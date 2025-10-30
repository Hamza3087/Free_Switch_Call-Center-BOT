#!/usr/bin/env python3
"""
Comprehensive Intent Detector with Extensive Phrase Matching
- Matches complete phrases instead of individual words
- Extensive phrase lists covering most variations
- Easy to maintain and extend phrase lists
- Reduces false positives from partial word matches
"""

import re
import time
import threading
import logging
import json
import os
from typing import Optional, Tuple, List, Dict

# Import config - adjust path as needed
try:
    from .config import MIN_INTENT_CONFIDENCE
except ImportError:
    MIN_INTENT_CONFIDENCE = 0.7  # Fallback default

class IntentDetector:
    def __init__(self, logger=None, hp_phrases=None):
        """Initialize intent detector with comprehensive phrase-based matching"""
        self.logger = logger or logging.getLogger(__name__)
        
        # Define complete phrases for each intent type
        # Extensive lists covering most variations
        self.intent_phrases = {
            "do_not_call": [
                # Basic do not call/contact
                "do not call",
                "don't call",
                "dont call",
                "do not call me",
                "don't call me",
                "dont call me",
                "do not contact",
                "don't contact",
                "dont contact",
                "do not contact me",
                "don't contact me",
                
                # Stop variations
                "stop calling",
                "stop calling me",
                "stop contacting",
                "stop contacting me",
                "quit calling",
                "quit calling me",
                "quit contacting",
                "cease calling",
                "cease contact",
                "cease and desist",
                "discontinue calling",
                "no more calls",
                "no more phone calls",
                "no further calls",
                "no further contact",
                
                # Please variations (polite)
                "please do not call",
                "please don't call",
                "please stop calling",
                "please do not contact",
                "please don't contact",
                "please stop contacting",
                "please no more calls",
                "please leave me alone",
                "kindly do not call",
                "kindly stop calling",
                
                # List management
                "remove me from your list",
                "take me off your list",
                "remove my number",
                "delete my number",
                "remove me from the list",
                "take me off the list",
                "remove from list",
                "take off list",
                "remove my phone number",
                "delete my phone number",
                "erase my number",
                "remove me from your calling list",
                "take me off your calling list",
                "remove me from your database",
                "delete me from your database",
                "remove my information",
                "delete my information",
                "unsubscribe",
                "unsubscribe me",
                "opt out",
                "opt me out",
                
                # Do not call list specific
                "put me on your do not call list",
                "add me to your do not call list",
                "put me on the do not call list",
                "add me to the do not call list",
                "put me on do not call",
                "add me to do not call",
                "place me on your do not call list",
                "i want to be on the do not call list",
                "register me on do not call",
                "dnc list",
                "add to dnc",
                "put on dnc",
                
                # Never/don't ever
                "never call again",
                "never call me again",
                "never contact again",
                "never contact me again",
                "don't ever call",
                "don't ever call me",
                "don't ever contact",
                "don't call again",
                "don't call me again",
                "don't call back",
                "don't call me back",
                "do not call again",
                "do not call back",
                "do not ever call",
                
                # Leave alone variations
                "leave me alone",
                "please leave me alone",
                "just leave me alone",
                "leave us alone",
                "let me alone",
                
                # Stop bothering/harassing
                "stop bothering me",
                "stop harassing me",
                "quit bothering me",
                "quit harassing me",
                "stop pestering me",
                "quit pestering me",
                "stop annoying me",
                "stop disturbing me",
                "don't bother me",
                "do not bother me",
                "don't disturb me",
                "do not disturb me",
                
                # This number variations
                "don't call this number",
                "do not call this number",
                "stop calling this number",
                "never call this number",
                "remove this number",
                "delete this number",
                "block this number",
                
                # Lose my number
                "lose my number",
                "forget my number",
                "throw away my number",
                "get rid of my number",
                
                # Fed up variations
                "i'm tired of these calls",
                "tired of you calling",
                "sick of these calls",
                "sick of you calling",
                "fed up with these calls",
                "had enough of these calls",
                "enough with the calls",
                "no more of these calls",
                
                # Wrong number claims (often used to stop calls)
                "wrong number stop calling",
                "you have the wrong number stop",
                "wrong person stop calling",
                
                # Business/work related
                "this is a business line",
                "this is a work number",
                "stop calling my business",
                "stop calling my work",
                "do not call my office",
                
                # Legal threats
                "i'll report you",
                "i will report you",
                "reporting to authorities",
                "calling my lawyer",
                "contacting my attorney",
                "this is harassment",
                "i'll sue",
                "i will sue",
            ],
            
            "not_interested": [
                # Basic not interested
                "not interested",
                "i'm not interested",
                "i am not interested",
                "we're not interested",
                "we are not interested",
                "he's not interested",
                "she's not interested",
                "not interested at all",
                "absolutely not interested",
                "definitely not interested",
                "really not interested",
                "just not interested",
                "simply not interested",
                
                # No interest
                "no interest",
                "no interest whatsoever",
                "zero interest",
                "have no interest",
                "i have no interest",
                "we have no interest",
                
                # No thanks variations
                "no thanks",
                "no thank you",
                "nope thanks",
                "no thanks though",
                "no thank you very much",
                "thanks but no thanks",
                "thank you but no",
                "thanks but no",
                "thanks anyway",
                "thanks anyways",
                "thank you anyway",
                "appreciate it but no",
                "i appreciate it but no",
                
                # Not for me/us
                "not for me",
                "not for us",
                "it's not for me",
                "it's not for us",
                "this isn't for me",
                "this isn't for us",
                "that's not for me",
                "that's not for us",
                
                # Don't want variations
                "don't want",
                "dont want",
                "i don't want",
                "i dont want",
                "we don't want",
                "we dont want",
                "do not want",
                "i do not want",
                "we do not want",
                "don't want it",
                "don't want this",
                "don't want that",
                "don't want any",
                "don't want anything",
                "want nothing",
                "i want nothing",
                
                # Don't need variations
                "don't need",
                "dont need",
                "i don't need",
                "i dont need",
                "we don't need",
                "we dont need",
                "do not need",
                "i do not need",
                "we do not need",
                "don't need it",
                "don't need this",
                "don't need that",
                "don't need any",
                "don't need anything",
                "need nothing",
                "i need nothing",
                "no need",
                "there's no need",
                
                # Don't care variations
                "don't care",
                "i don't care",
                "we don't care",
                "could not care less",
                "couldn't care less",
                "not bothered",
                "i'm not bothered",
                
                # Not looking variations
                "not looking",
                "i'm not looking",
                "i am not looking",
                "we're not looking",
                "we are not looking",
                "not looking for anything",
                "not looking to buy",
                "not looking to purchase",
                "not looking to switch",
                "not looking to change",
                "not in the market",
                "not in the market for",
                "we're not in the market",
                
                # Not buying/purchasing
                "not buying",
                "not buying anything",
                "i'm not buying",
                "we're not buying",
                "not purchasing",
                "not making purchases",
                "won't buy",
                "won't purchase",
                
                # Already have variations
                "already have",
                "i already have",
                "we already have",
                "already have it",
                "already have one",
                "already have that",
                "already have coverage",
                "already have insurance",
                "already have a plan",
                "already have a provider",
                "already have someone",
                "already covered",
                "i'm already covered",
                "we're already covered",
                "already taken care of",
                "already sorted",
                "already set",
                "already set up",
                "already signed up",
                "already enrolled",
                
                "all set",
                "i'm all set",
                "we're all set",
                "all good here",
                "i'm all good here",
                "we're all good here",
                # Removed ambiguous phrases that could be greeting/status responses:
                # "i'm good", "we're good", "i'm fine", "we're fine", 
                # "i'm okay", "we're okay", "doing fine", "doing okay"
                # "good to go", "i'm good to go"
                # These could be responses to "How are you?" or "Are you ready?" rather than rejections
                
                # Happy with current
                "happy with what i have",
                "happy with current",
                "happy with my current",
                "satisfied with current",
                "satisfied with what i have",
                "content with current",
                "fine with what i have",
                "good with what i have",
                "staying with current",
                "keeping what i have",
                "sticking with what i have",
                
                # Pass/skip variations
                "i'll pass",
                "we'll pass",
                "i pass",
                "we pass",
                "gonna pass",
                "going to pass",
                "have to pass",
                "i'll skip",
                "we'll skip",
                "skip it",
                "i'll skip it",
                
                # Not now/timing
                "not now",
                "not right now",
                "not at this time",
                "not at the moment",
                "not today",
                "not this time",
                "maybe later",
                "maybe another time",
                "perhaps later",
                "perhaps another time",
                "some other time",
                "another time",
                "bad time",
                "bad timing",
                "not a good time",
                "terrible timing",
                "wrong time",
                # "busy" alone removed - too ambiguous
                "i'm busy",
                "we're busy",
                "too busy",
                "i'm too busy",
                "busy right now",
                "in the middle of something",
                "can't talk",
                "can't talk now",
                "can't talk right now",
                
                # Not qualified/eligible
                "not qualified",
                "don't qualify",
                "i don't qualify",
                "we don't qualify",
                "not eligible",
                "i'm not eligible",
                "we're not eligible",
                "doesn't apply to me",
                "doesn't apply to us",
                "not applicable",
                "not relevant",
                "not relevant to me",
                
                # Age-related
                "too young",
                "i'm too young",
                "too old",
                "i'm too old",
                "not old enough",
                "not the right age",
                
                # Financial
                "can't afford",
                "cannot afford",
                "can't afford it",
                "no money",
                "no budget",
                "not in budget",
                "too expensive",
                "costs too much",
                "out of my budget",
                "financially unable",
                "economic situation",
                "times are tough",
                "money is tight",
                
                # Someone else handles
                "husband handles",
                "wife handles",
                "spouse handles",
                "partner handles",
                "someone else handles",
                "talk to my husband",
                "talk to my wife",
                "talk to my spouse",
                "not my decision",
                "not the decision maker",
                "can't make that decision",
                "need to consult",
                "need to think about it",
                "need to discuss",
                
                # Moving/leaving
                "i'm moving",
                "we're moving",
                "moving soon",
                "moving away",
                "leaving the area",
                "leaving the country",
                "leaving the state",
                "relocating",
                "going overseas",
                
                # Health/medical related
                "health issues",
                "medical issues",
                "i'm sick",
                "not well",
                "in the hospital",
                "health problems",
                "dealing with illness",
                "family emergency",
                "personal emergency",
                
                # Other provider/solution
                "have another provider",
                "use someone else",
                "go through someone else",
                "different company",
                "different provider",
                "competitor",
                "use your competitor",
                "with another company",
                
                # Just browsing/curious
                "just browsing",
                "just looking",
                "just curious",
                "just checking",
                "only browsing",
                "only looking",
                "window shopping",
                "just seeing what's out there",
            ],
            
            "callback": [
                # Basic callback requests
                "call me back",
                "call back",
                "call me back later",
                "call back later",
                "please call me back",
                "please call back",
                "can you call me back",
                "could you call me back",
                "call me back please",
                "call back please",
                
                # Try again variations
                "try again later",
                "try calling later",
                "try me later",
                "try back later",
                "call again later",
                "call me again later",
                "please try again later",
                "please call again later",
                "try again tomorrow",
                "call back tomorrow",
                "try tomorrow",
                
                # Better time variations
                "call at a better time",
                "better time to call",
                "call when convenient",
                "different time",
                "another time",
                "some other time",
                "maybe another time",
                "perhaps another time",
                "maybe later",
                "perhaps later",
                
                # Specific timing requests
                "call this evening",
                "call tonight",
                "call in the morning",
                "call this afternoon",
                "call next week",
                "call me next week",
                "call me tonight",
                "call me this evening",
                "call me in the morning",
                "call me this afternoon",
                
                # Reschedule variations
                "reschedule",
                "reschedule this call",
                "can we reschedule",
                "could we reschedule",
                "need to reschedule",
                "let's reschedule",
                
                # Not right now but willing
                "not right now but",
                "not now but later",
                "bad timing but",
                "wrong time but",
                "not a good time but",
                "busy right now but",
                "can't talk now but",
                "not available now but",
                
                # Will be available
                "i'll be available",
                "will be free",
                "will have time",
                "free later",
                "available later",
                "free this evening",
                "available this evening",
                "free tomorrow",
                "available tomorrow",
            ],
            
            "obscenity": [
                # F-word variations
                "fuck off",
                "fuck you",
                "go fuck yourself",
                "fuck this",
                "fuck that",
                "fucking stop",
                "fucking quit",
                "fucking leave me alone",
                "fucking annoying",
                "shut the fuck up",
                "what the fuck",
                "get fucked",
                "fuck no",
                "hell no fuck off",
                "fuck this shit",
                "fucking hell",
                "for fuck's sake",
                "for fucks sake",

                "motherfucker"
                "mother fucker"
                # Screw variations
                "screw you",
                "screw off",
                "screw this",
                "go screw yourself",
                
                # Piss variations
                "piss off",
                "pissed off",
                "pissing me off",
                "you're pissing me off",
                
                # Hell variations
                "go to hell",
                "burn in hell",
                "what the hell",
                "shut the hell up",
                "get the hell away",
                "get the hell out",
                "leave me the hell alone",
                "who the hell",
                "where the hell",
                "why the hell",
                
                # Damn variations
                "damn you",
                "god damn",
                "goddamn",
                "god damn it",
                "goddamnit",
                "damn it",
                "dammit",
                
                # Shit variations
                "this is shit",
                "you're shit",
                "full of shit",
                "bullshit",
                "this is bullshit",
                "horse shit",
                "horseshit",
                "cut the shit",
                "piece of shit",
                "shit head",
                "dipshit",
                
                # Ass variations
                "kiss my ass",
                "bite my ass",
                "pain in the ass",
                "get your ass",
                "shove it up your ass",
                "asshole",
                "you're an asshole",
                "dumbass",
                "jackass",
                "smart ass",
                "smartass",
                
                # Other offensive
                "bite me",
                "blow me",
                "suck it",
                "stick it",
                "shove it",
                "shove off",
                "drop dead",
                "eat shit",
                "son of a bitch",
                "bastard",
                "you bastard",
                "bitch",
                "son of a bitch",
                
                # Milder but still negative
                "shut up",
                "shut your mouth",
                "zip it",
                "be quiet",
                "stop talking",
                "you suck",
                "this sucks",
                "freaking annoying",
                "frigging annoying",
                
                # Idiot/stupid variations
                "you idiot",
                "you're an idiot",
                "fucking idiot",
                "stupid",
                "you're stupid",
                "moron",
                "you moron",
                "imbecile",
                "retard",
                "dumb fuck",
                "stupid fuck",
                
                # Get lost variations
                "get lost",
                "get out",
                "get away",
                "bugger off",
                "bog off",
                "sod off",
                "naff off",
                
                # Threatening
                "i'll kick your ass",
                "i'll beat your ass",
                "come here and say that",
                "i'll find you",
                "watch your back",
                
                # Waste of time
                "wasting my time",
                "waste of time",
                "stop wasting my time",
                "time waster",
                "fucking time waster",
                
                # Spam/scam accusations
                "fucking spammer",
                "fucking scammer",
                "scam artist",
                "con artist",
                "fucking thieves",
                "criminals",
                "fucking criminals",
            ]
        }
        
        # Additional word-boundary patterns for edge cases
        # These will be checked if no exact phrase matches
        self.boundary_patterns = {
            "do_not_call": [
                r"\bdo\s+not\s+call\b",
                r"\bdon'?t\s+call\b",
                r"\bstop\s+call(?:ing)?\b",
                r"\bremove\s+(?:me|my\s+number|this\s+number)\b",
                r"\bdo\s+not\s+contact\b",
                r"\badd\s+(?:me\s+)?to\s+(?:the\s+)?(?:do\s+not\s+call|dnc)\b",
                r"\btake\s+(?:me\s+)?off\s+(?:the\s+)?(?:list|calling\s+list)\b",
                r"\bnever\s+call\s+(?:me\s+)?again\b",
                r"\bleave\s+(?:me|us)\s+alone\b"
            ],
            "not_interested": [
                r"\bnot\s+interested\b",
                r"\bno\s+interest\b",
                r"\bdon'?t\s+(?:want|need|care)\b",
                r"\bnot\s+looking\b",
                r"\balready\s+have\b",
                r"\bi'?m?\s+all\s+set\b",  # Only match "all set", not "good/fine/okay"
                r"\bhappy\s+with\s+(?:my\s+)?current\b",
                r"\bi'?ll?\s+pass\b",
                r"\bnot\s+(?:now|right\s+now|at\s+this\s+time|today)\b",
                r"\bcan'?t\s+afford\b",
                r"\b(?:husband|wife|spouse)\s+handles\b",
                 r"\bnow\b"
            ],
            "callback": [
                r"\bcall\s+(?:me\s+)?back(?:\s+later)?\b",
                r"\btry\s+(?:again|calling)\s+later\b",
                r"\bcall\s+(?:me\s+)?(?:again\s+)?later\b",
                r"\breschedule(?:\s+this\s+call)?\b",
                r"\bbetter\s+time\s+to\s+call\b",
                r"\bcall\s+at\s+a\s+better\s+time\b",
                r"\bnot\s+(?:right\s+)?now\s+but(?:\s+later)?\b",
                r"\bbad\s+timing\s+but\b",
                r"\bbusy\s+(?:right\s+)?now\s+but\b",
                r"\bfree\s+later\b",
                r"\bavailable\s+later\b",
                r"\bcall\s+(?:me\s+)?(?:tonight|this\s+evening|tomorrow)\b"
            ],
            "obscenity": [
                r"\bfuck(?:ing)?\s+(?:off|you|this|that)\b",
                r"\bscrew\s+(?:off|you|this)\b",
                r"\bpiss(?:ed)?\s+off\b",
                r"\bgo\s+to\s+hell\b",
                r"\bshut\s+(?:the\s+)?(?:fuck|hell)\s+up\b",
                r"\bkiss\s+my\s+ass\b",
                r"\b(?:bull)?shit\b",
                r"\bass(?:hole)?\b",
                r"\bget\s+(?:lost|fucked|out)\b",
            ]
        }
        
        # Compile boundary patterns
        self.compiled_patterns = {}
        
        # Override keywords that take absolute priority
        self.override_keywords = {
            "hold_press": hp_phrases if hp_phrases is not None else []
        }
        
        # Log honeypot phrases loaded
        hp_list = self.override_keywords.get("hold_press", [])
        if hp_list:
            self.logger.info(f"🍯 IntentDetector loaded {len(hp_list)} honeypot phrases: {hp_list}")
        else:
            self.logger.info("🍯 IntentDetector loaded with no honeypot phrases")
        for intent_type, patterns in self.boundary_patterns.items():
            self.compiled_patterns[intent_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # State tracking
        self.negative_detected = False
        self.negative_intent = None
        self.detected_intents = []
        
        # Confidence thresholds
        self.partial_threshold = 0.55
        self.final_threshold = MIN_INTENT_CONFIDENCE
        
        self.logger.info(f"Intent detector initialized with {sum(len(phrases) for phrases in self.intent_phrases.values())} phrases")
    
    
    def detect_intent(self, text: str, is_partial: bool = False) -> Optional[Tuple[str, float]]:
        """
        Detect intents using exact phrase matching first, then patterns
        
        Args:
            text: User's speech text
            is_partial: Whether this is partial (incomplete) speech
            
        Returns:
            Tuple of (intent_name, confidence) or None
        """
        if not text:
            return None
            
        # Normalize text for matching
        text_lower = text.lower().strip()
        
        # Remove common filler words for better matching
        text_cleaned = self._clean_text(text_lower)
        
        # FIRST: Check override keywords (highest priority)
        override_result = self._check_override_keywords(text_lower, text_cleaned)
        if override_result:
            intent_type, confidence = override_result
            threshold = self.partial_threshold if is_partial else self.final_threshold
            
            if confidence >= threshold:
                self.negative_detected = True
                self.negative_intent = intent_type
                self.logger.info(f"Override keyword detected: {intent_type} (confidence: {confidence:.2f})")
                return (intent_type, confidence)
        
        # Second, check for exact phrase matches
        intent_result = self._check_exact_phrases(text_lower, text_cleaned)
        
        if intent_result:
            intent_type, confidence = intent_result
            
            # Boost confidence for call-ending intents
            if intent_type in ["do_not_call", "callback", "hold_press"]:
                confidence = min(confidence + 0.1, 0.95)
            
            # Check against threshold
            threshold = self.partial_threshold if is_partial else self.final_threshold
            
            if confidence >= threshold:
                # Mark as negative intent for call-ending intents
                if intent_type in ["do_not_call", "not_interested", "obscenity", "callback", "hold_press"]:
                    self.negative_detected = True
                    self.negative_intent = intent_type
                self.logger.info(f"Intent detected: {intent_type} (confidence: {confidence:.2f})")
                return (intent_type, confidence)
        
        # If no exact match, check boundary patterns (more flexible)
        intent_result = self._check_boundary_patterns(text_lower)
        
        if intent_result:
            intent_type, confidence = intent_result
            
            # Lower confidence for pattern matches vs exact phrases
            confidence *= 0.9
            
            threshold = self.partial_threshold if is_partial else self.final_threshold
            
            if confidence >= threshold:
                # Mark as negative intent for call-ending intents
                if intent_type in ["do_not_call", "not_interested", "obscenity", "callback", "hold_press"]:
                    self.negative_detected = True
                    self.negative_intent = intent_type
                self.logger.info(f"Intent detected (pattern): {intent_type} (confidence: {confidence:.2f})")
                return (intent_type, confidence)
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing common filler words
        """
        # Remove common fillers that don't affect meaning
        fillers = [
            "um", "uh", "uhm", "er", "ah", "like",
            "you know", "i mean", "actually", "basically",
            "please", "just", "really", "very", "well",
            "so", "okay", "alright", "right"
        ]
        
        cleaned = text
        for filler in fillers:
            cleaned = re.sub(r'\b' + re.escape(filler) + r'\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

    
    def _check_override_keywords(self, text: str, text_cleaned: str) -> Optional[Tuple[str, float]]:
        """
        Check for high-priority override keywords that take absolute precedence
        
        Args:
            text: Original normalized text
            text_cleaned: Text with filler words removed
            
        Returns:
            Tuple of (intent_name, confidence) or None
        """
        best_match = None
        best_confidence = 0
        
        for intent_type, keywords in self.override_keywords.items():
            for keyword in keywords:
                # Check if keyword appears as a word boundary in either text version
                if (re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE) or
                    re.search(r'\b' + re.escape(keyword) + r'\b', text_cleaned, re.IGNORECASE)):
                    
                    # High confidence for override keywords (they take priority)
                    confidence = 0.95
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_type
                    
                    self.logger.debug(f"Override keyword match: '{keyword}' -> {intent_type} (confidence: {confidence:.2f})")
        
        if best_match:
            return (best_match, best_confidence)
        
        return None
    
    def _check_exact_phrases(self, text: str, text_cleaned: str) -> Optional[Tuple[str, float]]:
        """
        Check for exact phrase matches
        """
        best_match = None
        best_confidence = 0
        
        for intent_type, phrases in self.intent_phrases.items():
            for phrase in phrases:
                # Check both original and cleaned text
                if phrase in text or phrase in text_cleaned:
                    # Calculate confidence based on how much of the text is the phrase
                    phrase_ratio = len(phrase) / max(len(text), 1)
                    
                    # Higher confidence for exact matches
                    if text == phrase or text_cleaned == phrase:
                        confidence = 0.95
                    else:
                        # Confidence based on phrase coverage
                        confidence = 0.7 + (phrase_ratio * 0.2)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_type
                    
                    self.logger.debug(f"Phrase match: '{phrase}' in '{text}' (confidence: {confidence:.2f})")
        
        if best_match:
            return (best_match, best_confidence)
        
        return None
    
    def _check_boundary_patterns(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Check using regex patterns with word boundaries
        """
        best_match = None
        best_confidence = 0
        
        for intent_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    # Calculate confidence based on match coverage
                    match_text = match.group()
                    match_ratio = len(match_text) / max(len(text), 1)
                    
                    # Base confidence for pattern matches
                    confidence = 0.6 + (match_ratio * 0.2)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_type
                    
                    self.logger.debug(f"Pattern match: {pattern.pattern} matched '{match_text}' (confidence: {confidence:.2f})")
        
        if best_match:
            return (best_match, best_confidence)
        
        return None
    
    def check_transcript_for_intents(self, transcript: List[Dict]) -> Optional[Tuple[str, float]]:
        """
        Check entire transcript for negative intents
        """
        negative_intent = None
        max_confidence = 0
        
        for speech_item in transcript:
            text = speech_item.get('text', '')
            is_partial = speech_item.get('type') == 'PARTIAL'
            
            intent_result = self.detect_intent(text, is_partial=is_partial)
            
            if intent_result and intent_result[1] > max_confidence:
                negative_intent = intent_result[0]
                max_confidence = intent_result[1]
        
        if max_confidence >= self.final_threshold:
            self.negative_detected = True
            self.negative_intent = negative_intent
            return (negative_intent, max_confidence)
        
        return None
    
    def add_phrase(self, intent_type: str, phrase: str):
        """
        Dynamically add a new phrase to an intent type
        
        Args:
            intent_type: The intent category ('do_not_call', 'not_interested', 'obscenity')
            phrase: The phrase to add
        """
        if intent_type in self.intent_phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower not in self.intent_phrases[intent_type]:
                self.intent_phrases[intent_type].append(phrase_lower)
                self.logger.info(f"Added phrase '{phrase}' to {intent_type}")
        else:
            self.logger.warning(f"Unknown intent type: {intent_type}")
    
    def remove_phrase(self, intent_type: str, phrase: str):
        """
        Remove a phrase from an intent type
        
        Args:
            intent_type: The intent category
            phrase: The phrase to remove
        """
        if intent_type in self.intent_phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower in self.intent_phrases[intent_type]:
                self.intent_phrases[intent_type].remove(phrase_lower)
                self.logger.info(f"Removed phrase '{phrase}' from {intent_type}")
    
    def get_phrases(self, intent_type: str = None) -> Dict[str, List[str]]:
        """
        Get current phrases for debugging/monitoring
        
        Args:
            intent_type: Specific intent type or None for all
            
        Returns:
            Dictionary of intent phrases
        """
        if intent_type:
            return {intent_type: self.intent_phrases.get(intent_type, [])}
        return self.intent_phrases.copy()
    
    def reset(self):
        """Reset the detector state"""
        self.detected_intents = []
        self.negative_detected = False
        self.negative_intent = None
        self.logger.debug("Intent detector reset")
    
    def get_stats(self) -> Dict:
        """Get statistics about the detector"""
        return {
            'total_phrases': sum(len(phrases) for phrases in self.intent_phrases.values()),
            'phrases_per_intent': {k: len(v) for k, v in self.intent_phrases.items()},
            'negative_detected': self.negative_detected,
            'negative_intent': self.negative_intent,
            'detected_intents': self.detected_intents
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create detector
    detector = IntentDetector()
    
    # Print statistics
    stats = detector.get_stats()
    print(f"Intent Detector loaded with {stats['total_phrases']} total phrases")
    print(f"Phrases per intent: {stats['phrases_per_intent']}")
    print("-" * 50)
    
    # Test cases
    test_cases = [
        # Should match do_not_call
        ("Please do not call me again", "do_not_call"),
        ("stop calling", "do_not_call"),
        ("remove me from your list", "do_not_call"),
        ("put me on the do not call list", "do_not_call"),
        
        # Should match not_interested
        ("I'm not interested", "not_interested"),
        ("no thanks", "not_interested"),
        ("already have insurance", "not_interested"),
        ("my husband handles all that", "not_interested"),
        
        # Should match obscenity
        ("fuck off", "obscenity"),
        ("go to hell", "obscenity"),
        
        # Should NOT match (individual words)
        ("I do not want to call you", None),
        ("That's interesting but not for me", None),
        ("I'll call you back", None),
    ]
    
    print("\nTesting Intent Detector")
    print("-" * 50)
    
    for text, expected in test_cases:
        result = detector.detect_intent(text)
        
        if result:
            intent, confidence = result
            status = "✅" if intent == expected else "❌"
            print(f"{status} '{text[:50]}...' if len(text) > 50 else '{text}'")
            print(f"   → {intent} (confidence: {confidence:.2f})")
        else:
            status = "✅" if expected is None else "❌"
            print(f"{status} '{text[:50]}...' if len(text) > 50 else '{text}'")
            print(f"   → No intent detected")
        
        detector.reset()  # Reset between tests