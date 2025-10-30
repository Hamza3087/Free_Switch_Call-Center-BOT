#!/bin/bash

################################################################################
# Modular SIP Bot Setup Script for Ubuntu 24.04 (No CUDA/NVIDIA)
# 
# Features:
# - Modular design with state tracking
# - Can resume from last successful module
# - Individual module execution
# - Verification after each module
# - Works on both fresh and existing installations
#
# Key Changes:
# - Removed all CUDA and NVIDIA driver installation components
# - Removed GPU-related environment variables
# - Simplified to focus on SIP/PJSIP and Python components only
# - Added Module 6: Disable Auto Updates to prevent SIP interruptions
#
# Usage:
# ./setup_sip_bot_modular.sh              # Run all modules
# ./setup_sip_bot_modular.sh --status     # Check installation status
# ./setup_sip_bot_modular.sh --module 3   # Run specific module
# ./setup_sip_bot_modular.sh --resume     # Resume from last failure
# ./setup_sip_bot_modular.sh --verify     # Verify all installations
#
################################################################################

set -e  # Exit on error

# Configuration
STATE_FILE="/root/.sip_bot_setup_state"
LOG_FILE="/root/sip_bot_setup.log"
PJSIP_VERSION="2.15.1"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
    log "INFO: $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

print_module() {
    echo -e "${BLUE}[MODULE]${NC} $1"
    log "MODULE: $1"
}

# State management
save_state() {
    local module=$1
    local status=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create state file if it doesn't exist
    [ ! -f "$STATE_FILE" ] && touch "$STATE_FILE"
    
    # Update or add module state
    if grep -q "^MODULE_$module=" "$STATE_FILE"; then
        sed -i "s/^MODULE_$module=.*/MODULE_$module=$status|$timestamp/" "$STATE_FILE"
    else
        echo "MODULE_$module=$status|$timestamp" >> "$STATE_FILE"
    fi
}

get_state() {
    local module=$1
    if [ -f "$STATE_FILE" ]; then
        grep "^MODULE_$module=" "$STATE_FILE" 2>/dev/null | cut -d'=' -f2 | cut -d'|' -f1 || echo "NOT_RUN"
    else
        echo "NOT_RUN"
    fi
}

get_all_states() {
    echo -e "\n${BLUE}=== Installation Status ===${NC}"
    echo "Module 1 (Essential Tools): $(get_state 1)"
    echo "Module 2 (PJSIP): $(get_state 2)"
    echo "Module 3 (System Config): $(get_state 3)"
    echo "Module 4 (SIP Bot): $(get_state 4)"
    echo "Module 5 (Systemd Service): $(get_state 5)"
    echo "Module 6 (Disable Auto Updates): $(get_state 6)"
    echo ""
}

# Verification functions
verify_module_1() {
    print_status "Verifying essential tools..."
    local status="SUCCESS"
    
    for tool in wget curl unzip zip ffmpeg; do
        if ! command -v $tool &> /dev/null; then
            print_error "$tool is not installed"
            status="FAILED"
        else
            print_status "✓ $tool is installed"
        fi
    done
    
    return $([ "$status" == "SUCCESS" ] && echo 0 || echo 1)
}

verify_module_2() {
    print_status "Verifying PJSIP installation..."
    
    # Check if PJSIP libraries exist
    if [ ! -d "/root/pjproject-$PJSIP_VERSION/pjsip/lib" ]; then
        print_error "PJSIP libraries not found"
        return 1
    fi
    
    # Check for any PJSUA binary (path varies by build)
    if find /root/pjproject-$PJSIP_VERSION -name "pjsua*" -type f -executable | grep -q .; then
        print_status "✓ PJSIP binaries found"
    else
        print_warning "PJSIP binary not found (not critical for Python bindings)"
    fi
    
    # Check epoll compilation
    if find /root/pjproject-$PJSIP_VERSION -name "ioqueue_epoll.o" | grep -q .; then
        print_status "✓ PJSIP compiled with epoll support"
    else
        print_error "PJSIP epoll support not found"
        return 1
    fi
    
    # Check Python bindings in system Python
    if python3 -c "import pjsua2" 2>/dev/null; then
        print_status "✓ PJSUA2 Python bindings installed in system Python"
    else
        print_warning "PJSUA2 not found in system Python - will try to install"
        
        # Try to install if source exists
        if [ -d "/root/pjproject-$PJSIP_VERSION/pjsip-apps/src/swig/python" ]; then
            print_status "Installing PJSUA2 in system Python..."
            cd /root/pjproject-$PJSIP_VERSION/pjsip-apps/src/swig/python
            python3 setup.py install 2>/dev/null || true
        fi
    fi
    
    # Check Python bindings in venv if it exists (optional)
    if [ -f "/root/venv/bin/python" ]; then
        if /root/venv/bin/python -c "import pjsua2" 2>/dev/null; then
            print_status "✓ PJSUA2 Python bindings also installed in venv"
        else
            print_warning "PJSUA2 not in venv - installing..."
            cd /root/pjproject-$PJSIP_VERSION/pjsip-apps/src/swig/python
            /root/venv/bin/python setup.py install 2>/dev/null || true
        fi
    fi
    
    # Final check - at least system Python should work
    if python3 -c "import pjsua2" 2>/dev/null; then
        print_status "✓ PJSIP module 2 verified successfully"
        return 0
    else
        print_error "PJSUA2 Python bindings could not be verified"
        return 1
    fi
}

verify_module_3() {
    print_status "Verifying system configuration..."
    
    local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
    local wmem_max=$(sysctl -n net.core.wmem_max 2>/dev/null)
    local file_max=$(sysctl -n fs.file-max 2>/dev/null)
    
    if [ "$rmem_max" -ge "26214400" ] && [ "$wmem_max" -ge "26214400" ]; then
        print_status "✓ Network buffers configured: rmem=$rmem_max, wmem=$wmem_max"
    else
        print_warning "Network buffers not optimal: rmem=$rmem_max, wmem=$wmem_max"
    fi
    
    if [ "$file_max" -ge "2097152" ]; then
        print_status "✓ File limits configured: $file_max"
    else
        print_warning "File limits not optimal: $file_max"
    fi
    
    return 0
}

verify_module_4() {
    print_status "Verifying SIP Bot installation..."
    
    if [ ! -d "/root/sip-bot" ]; then
        print_error "SIP Bot directory not found"
        return 1
    fi
    
    if [ ! -f "/root/venv/bin/python" ]; then
        print_error "Python venv not found"
        return 1
    fi
    
    print_status "✓ SIP Bot directory and venv exist"
    return 0
}

verify_module_5() {
    print_status "Verifying systemd service..."
    
    if [ ! -f "/etc/systemd/system/sip_bot.service" ]; then
        print_error "Systemd service file not found"
        return 1
    fi
    
    if systemctl is-enabled sip_bot &> /dev/null; then
        print_status "✓ SIP Bot service is enabled"
    else
        print_warning "SIP Bot service exists but is not enabled"
    fi
    
    return 0
}

verify_module_6() {
    print_status "Verifying auto-updates are disabled..."
    
    local status="SUCCESS"
    
    # Check unattended-upgrades service
    if systemctl is-enabled unattended-upgrades &> /dev/null; then
        print_error "unattended-upgrades service is still enabled"
        status="FAILED"
    else
        print_status "✓ unattended-upgrades service is disabled"
    fi
    
    # Check apt-daily timers
    if systemctl is-active apt-daily.timer &> /dev/null; then
        print_error "apt-daily.timer is still active"
        status="FAILED"
    else
        print_status "✓ apt-daily.timer is disabled"
    fi
    
    if systemctl is-active apt-daily-upgrade.timer &> /dev/null; then
        print_error "apt-daily-upgrade.timer is still active"
        status="FAILED"
    else
        print_status "✓ apt-daily-upgrade.timer is disabled"
    fi
    
    # Check 20auto-upgrades configuration
    if [ -f "/etc/apt/apt.conf.d/20auto-upgrades" ]; then
        if grep -q 'APT::Periodic::Update-Package-Lists "0"' /etc/apt/apt.conf.d/20auto-upgrades && \
           grep -q 'APT::Periodic::Unattended-Upgrade "0"' /etc/apt/apt.conf.d/20auto-upgrades; then
            print_status "✓ Auto-update configuration is disabled"
        else
            print_warning "Auto-update configuration file exists but may not be properly configured"
        fi
    else
        print_warning "Auto-update configuration file not found"
    fi
    
    return $([ "$status" == "SUCCESS" ] && echo 0 || echo 1)
}

# Module 1: Essential Tools
module_1_essential_tools() {
    print_module "Module 1: Installing Essential Tools"
    
    if [ "$(get_state 1)" == "SUCCESS" ]; then
        print_warning "Module 1 already completed. Skipping..."
        return 0
    fi
    
    apt update
    apt install -y wget curl unzip zip ffmpeg
    apt install -y build-essential software-properties-common linux-headers-$(uname -r)
    apt install -y python3.12 python3.12-venv python3.12-dev python3-pip
    
    if verify_module_1; then
        save_state 1 "SUCCESS"
        print_status "Module 1 completed successfully"
        return 0
    else
        save_state 1 "FAILED"
        return 1
    fi
}

# Module 2: PJSIP
module_2_pjsip() {
    print_module "Module 2: Installing PJSIP $PJSIP_VERSION"
    
    if [ "$(get_state 2)" == "SUCCESS" ]; then
        if verify_module_2; then
            print_warning "Module 2 already completed. Checking venv..."
            
            # Still need to ensure it's in venv if venv exists
            if [ -f "/root/venv/bin/python" ]; then
                if ! /root/venv/bin/python -c "import pjsua2" 2>/dev/null; then
                    print_status "Installing PJSUA2 in existing venv..."
                    cd /root/pjproject-$PJSIP_VERSION/pjsip-apps/src/swig/python
                    /root/venv/bin/python setup.py install
                fi
            fi
            return 0
        fi
    fi
    
    # Install dependencies
    apt update
    apt install -y build-essential swig pkg-config libasound2-dev libpulse-dev \
        libssl-dev libsrtp2-dev libavcodec-dev libavformat-dev libavutil-dev \
        libswscale-dev libavdevice-dev
    
    cd ~
    
    # Download PJSIP if not present
    if [ ! -d "pjproject-$PJSIP_VERSION" ]; then
        wget https://github.com/pjsip/pjproject/archive/refs/tags/$PJSIP_VERSION.tar.gz -O pjproject.tar.gz
        tar -xzf pjproject.tar.gz
    fi
    
    cd ~/pjproject-$PJSIP_VERSION
    
    # Configure for high concurrency
    print_status "Configuring PJSIP..."
    cat > pjlib/include/pj/config_site.h << 'EOF'
/* config_site.h - Optimized for High-Concurrency SIP Bot */
#define PJ_LINUX_HAS_EPOLL          1
#define PJ_HAS_EPOLL                1
#define PJ_IOQUEUE_DEFAULT_IMPL     PJ_IOQUEUE_EPOLL
#define PJ_IOQUEUE_HAS_SAFE_UNREG   1
#define PJ_IOQUEUE_MAX_HANDLES      65535
#define PJSIP_MAX_TSX_COUNT         (640*1024-1)
#define PJSIP_MAX_DIALOG_COUNT      (640*1024-1)
#define PJSUA_MAX_CALLS             10000
#define PJSUA_MAX_ACC               1000
#define PJSUA_MAX_PLAYERS           10000
#define PJSUA_MAX_RECORDERS         10000
#define PJSUA_MAX_CONF_PORTS        10000
#define PJ_THREAD_DEFAULT_STACK_SIZE    (8*1024*1024)
#define PJSIP_MAX_WORKERS           32
#define PJ_LOG_MAX_LEVEL           5
#define PJ_OS_HAS_CHECK_STACK      0
#define PJSIP_SAFE_MODULE          0
#define PJSIP_UNESCAPE_IN_PLACE    1
#define PJ_HASH_USE_OWN_TOLOWER    1
#define PJSIP_POOL_LEN_ENDPT       8000
#define PJSIP_POOL_INC_ENDPT       4000
#define PJSIP_POOL_RDATA_LEN       8000
#define PJSIP_POOL_RDATA_INC       4000
#define PJSIP_POOL_LEN_TDATA       8000
#define PJSIP_POOL_INC_TDATA       4000
#define PJMEDIA_SESSION_SIZE       8000
#define PJMEDIA_SESSION_INC        4000
#define __FD_SETSIZE               65535
#define FD_SETSIZE                 65535
#define PJMEDIA_MAX_WORKER_THREADS  16
#define PJMEDIA_CONF_USE_SWITCH_BOARD 0
#define PJMEDIA_SOUND_BUFFER_COUNT  16
EOF
    
    # Force epoll
    mv pjlib/src/pj/ioqueue_select.c pjlib/src/pj/ioqueue_select.c.BACKUP 2>/dev/null || true
    echo "/* Force epoll - select disabled */" > pjlib/src/pj/ioqueue_select.c
    
    # Compile
    print_status "Compiling PJSIP..."
    make distclean || true
    export CFLAGS="-O3 -march=native -fPIC -DNDEBUG"
    export CXXFLAGS="$CFLAGS"
    ./configure --enable-epoll --disable-video
    make dep
    make clean
    make -j$(nproc)
    
    # Build Python bindings
    print_status "Building Python bindings..."
    cd pjsip-apps/src/swig/python
    make clean
    rm -rf build/ dist/ *.egg-info
    make
    
    # Install in system Python
    python3 setup.py install
    
    # Install in venv if it exists
    if [ -f "/root/venv/bin/python" ]; then
        print_status "Installing PJSUA2 in venv..."
        /root/venv/bin/python setup.py install
    fi
    
    if verify_module_2; then
        save_state 2 "SUCCESS"
        print_status "Module 2 completed successfully"
        return 0
    else
        save_state 2 "FAILED"
        return 1
    fi
}

# Module 3: System Configuration
module_3_system_config() {
    print_module "Module 3: Configuring System Settings"
    
    if [ "$(get_state 3)" == "SUCCESS" ]; then
        if verify_module_3; then
            print_warning "Module 3 already completed. Skipping..."
            return 0
        fi
    fi
    
    # Check if already configured
    if grep -q "SIP Bot Network Tuning" /etc/sysctl.conf; then
        print_warning "System settings already configured. Updating..."
    else
        cat >> /etc/sysctl.conf << 'EOF'

# SIP Bot Network Tuning & Buffer Settings
fs.file-max = 2097152
fs.nr_open = 2097152
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65536
net.core.rmem_max = 26214400
net.core.wmem_max = 26214400
net.core.rmem_default = 2097152
net.core.wmem_default = 2097152
net.ipv4.tcp_max_syn_backlog = 65536
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30
net.ipv4.udp_rmem_min = 65536
net.ipv4.udp_wmem_min = 65536
EOF
    fi
    
    sysctl -p
    ulimit -n 65536
    
    if verify_module_3; then
        save_state 3 "SUCCESS"
        print_status "Module 3 completed successfully"
        return 0
    else
        save_state 3 "FAILED"
        return 1
    fi
}

# Module 4: SIP Bot Setup
module_4_sip_bot() {
    print_module "Module 4: Setting up SIP Bot"
    
    if [ "$(get_state 4)" == "SUCCESS" ]; then
        if verify_module_4; then
            print_warning "Module 4 already completed. Skipping..."
            return 0
        fi
    fi
    
    if [ ! -f "/root/sip-bot.zip" ]; then
        print_error "sip-bot.zip not found in /root/"
        print_warning "Please upload sip-bot.zip and re-run this module"
        save_state 4 "SKIPPED"
        return 1
    fi
    
    cd /root
    print_status "Extracting SIP bot..."
    unzip -o sip-bot.zip
    
    # Create venv if not exists
    if [ ! -f "/root/venv/bin/python" ]; then
        print_status "Creating Python virtual environment..."
        python3.12 -m venv /root/venv
    fi
    
    # Install requirements
    if [ -f "/root/sip-bot/requirements.txt" ]; then
        print_status "Installing Python requirements in venv..."
        /root/venv/bin/pip install --upgrade pip
        /root/venv/bin/pip install -r /root/sip-bot/requirements.txt
    fi
    
    # Install PJSUA2 in venv - check both venv and system
    print_status "Ensuring PJSUA2 is available..."
    
    # First check if it works in venv
    if /root/venv/bin/python -c "import pjsua2" 2>/dev/null; then
        print_status "✓ PJSUA2 already available in venv"
    elif python3 -c "import pjsua2" 2>/dev/null; then
        # System has it, try to install in venv
        print_status "PJSUA2 found in system Python, installing in venv..."
        if [ -d "/root/pjproject-$PJSIP_VERSION/pjsip-apps/src/swig/python" ]; then
            cd /root/pjproject-$PJSIP_VERSION/pjsip-apps/src/swig/python
            /root/venv/bin/python setup.py install 2>/dev/null || {
                print_warning "Could not install PJSUA2 in venv, will use system-wide installation"
                print_status "Note: Service can still work with system-wide PJSUA2"
            }
        fi
    else
        print_warning "PJSUA2 not found - Module 2 may need to be run"
    fi
    
    if verify_module_4; then
        save_state 4 "SUCCESS"
        print_status "Module 4 completed successfully"
        
        # Show status
        if /root/venv/bin/python -c "import pjsua2" 2>/dev/null; then
            print_status "✓ PJSUA2 available in venv"
        elif python3 -c "import pjsua2" 2>/dev/null; then
            print_status "✓ PJSUA2 available system-wide"
        else
            print_warning "PJSUA2 not available - may need to run Module 2"
        fi
        
        return 0
    else
        save_state 4 "FAILED"
        return 1
    fi
}

# Module 5: Systemd Service
module_5_systemd_service() {
    print_module "Module 5: Creating Systemd Service"
    
    if [ "$(get_state 5)" == "SUCCESS" ]; then
        if verify_module_5; then
            print_warning "Module 5 already completed. Skipping..."
            return 0
        fi
    fi
    
    cat > /etc/systemd/system/sip_bot.service << 'EOF'
[Unit]
Description=SuiteCRM SIP Bot Service (Running from /root/sip-bot)
After=network.target mysql.service
Wants=mysql.service

[Service]
User=root
Group=root
WorkingDirectory=/root/sip-bot
ExecStart=/root/venv/bin/python3 /root/sip-bot/sip_bot_server.py
ExecReload=/bin/kill -USR1 $MAINPID
Restart=on-failure
RestartSec=10

# Resource limits
TasksMax=76804
LimitNOFILE=65536
LimitNPROC=32768
LimitMEMLOCK=8388608
LimitSIGPENDING=256015
LimitMSGQUEUE=819200
LimitNICE=0
LimitRTPRIO=0

# Environment variables
Environment="PYTHONUNBUFFERED=1"
Environment="PYTHONPATH=/root/sip-bot"

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sip_bot

TimeoutStopSec=30
KillSignal=SIGTERM
KillMode=process

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable sip_bot.service
    
    if verify_module_5; then
        save_state 5 "SUCCESS"
        print_status "Module 5 completed successfully"
        return 0
    else
        save_state 5 "FAILED"
        return 1
    fi
}

# Module 6: Disable Auto Updates
module_6_disable_auto_updates() {
    print_module "Module 6: Disabling Automatic Updates (Prevents SIP Interruptions)"
    
    if [ "$(get_state 6)" == "SUCCESS" ]; then
        if verify_module_6; then
            print_warning "Module 6 already completed. Skipping..."
            return 0
        fi
    fi
    
    print_status "Stopping and disabling unattended-upgrades service..."
    systemctl stop unattended-upgrades 2>/dev/null || true
    systemctl disable unattended-upgrades 2>/dev/null || true
    
    print_status "Disabling apt-daily timers..."
    systemctl disable apt-daily.timer 2>/dev/null || true
    systemctl disable apt-daily-upgrade.timer 2>/dev/null || true
    systemctl stop apt-daily.timer 2>/dev/null || true
    systemctl stop apt-daily-upgrade.timer 2>/dev/null || true
    
    print_status "Configuring auto-update settings..."
    cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Download-Upgradeable-Packages "0";
APT::Periodic::AutocleanInterval "0";
APT::Periodic::Unattended-Upgrade "0";
EOF
    
    # Also disable auto-reboot if unattended-upgrades config exists
    if [ -f "/etc/apt/apt.conf.d/50unattended-upgrades" ]; then
        print_status "Disabling automatic reboots..."
        if grep -q "Unattended-Upgrade::Automatic-Reboot" /etc/apt/apt.conf.d/50unattended-upgrades; then
            sed -i 's/.*Unattended-Upgrade::Automatic-Reboot.*/Unattended-Upgrade::Automatic-Reboot "false";/' /etc/apt/apt.conf.d/50unattended-upgrades
        else
            echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades
        fi
    fi
    
    if verify_module_6; then
        save_state 6 "SUCCESS"
        print_status "Module 6 completed successfully"
        print_warning "Remember to manually update during maintenance windows!"
        return 0
    else
        save_state 6 "FAILED"
        return 1
    fi
}

# Main execution
main() {
    # Check root
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
    
    # Parse arguments
    case "$1" in
        --status)
            get_all_states
            exit 0
            ;;
        --verify)
            print_module "Running Full Verification"
            verify_module_1
            verify_module_2
            verify_module_3
            verify_module_4
            verify_module_5
            verify_module_6
            exit 0
            ;;
        --module)
            if [ -z "$2" ]; then
                print_error "Please specify module number (1-6)"
                exit 1
            fi
            case "$2" in
                1) module_1_essential_tools ;;
                2) module_2_pjsip ;;
                3) module_3_system_config ;;
                4) module_4_sip_bot ;;
                5) module_5_systemd_service ;;
                6) module_6_disable_auto_updates ;;
                *) print_error "Invalid module number: $2" ;;
            esac
            exit $?
            ;;
        --resume)
            print_status "Resuming from last incomplete module..."
            for i in {1..6}; do
                state=$(get_state $i)
                if [ "$state" != "SUCCESS" ]; then
                    print_status "Resuming from Module $i (status: $state)"
                    case "$i" in
                        1) module_1_essential_tools ;;
                        2) module_2_pjsip ;;
                        3) module_3_system_config ;;
                        4) module_4_sip_bot ;;
                        5) module_5_systemd_service ;;
                        6) module_6_disable_auto_updates ;;
                    esac
                    [ $? -ne 0 ] && exit 1
                fi
            done
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --status    Check installation status"
            echo "  --verify    Verify all installations"
            echo "  --module N  Run specific module (1-6)"
            echo "  --resume    Resume from last incomplete module"
            echo "  --help      Show this help"
            echo ""
            echo "Modules:"
            echo "  1: Essential Tools (wget, curl, unzip, ffmpeg, etc.)"
            echo "  2: PJSIP with Python bindings"
            echo "  3: System Configuration"
            echo "  4: SIP Bot Setup"
            echo "  5: Systemd Service"
            echo "  6: Disable Auto Updates (prevents SIP interruptions)"
            exit 0
            ;;
        *)
            # Run all modules
            print_status "Running complete installation..."
            module_1_essential_tools && \
            module_2_pjsip && \
            module_3_system_config && \
            module_4_sip_bot && \
            module_5_systemd_service && \
            module_6_disable_auto_updates
            ;;
    esac
    
    # Final summary
    echo ""
    get_all_states
    
    echo ""
    print_status "Setup complete! Next steps:"
    echo "1. Verify installation: $0 --verify"
    echo "2. Start service: systemctl start sip_bot"
    echo "3. Check logs: journalctl -u sip_bot -f"
    echo "4. Manual updates: Run 'apt update && apt upgrade' during maintenance windows"
}

# Run main
main "$@"