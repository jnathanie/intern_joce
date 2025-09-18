#!/usr/bin/env python3
"""
Terminal Interface for Car Recommendation System
Usage: python term.py

Make sure your Flask app (app.py) is running on localhost:5000 first!
"""

import requests
import json
import sys
import os
from typing import List, Dict, Any, Optional

# =============================================================================
# CONFIGURATION - API settings and display options
# =============================================================================

API_BASE_URL = "http://localhost:5000"  # Change if your Flask app runs on different host/port

# Terminal colors for better output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Available options based on app.py scoring values
SEGMENTATION_OPTIONS = [
    "Agriculture, Forestry & Fishing",
    "Accommodation", 
    "Construction",
    "Courier",
    "Distributor & Retail",
    "Education"
]

TIPE_JALAN_OPTIONS = [
    "Off-road",
    "On-road Datar", 
    "On-road Perbukitan"
]

TONNASE_OPTIONS = [
    "<5 ton (Pickup, LCV)",
    "5 - 7 Ton (4 Ban)",
    "8 - 15 Ton (6 Ban)",
    "16 - 23 Ton",
    "23 - 34 Ton", 
    ">35 Ton"
]

KUBIKASI_OPTIONS = [
    "<12 M3",
    "13 - 17 M3 (4 Ban Long)",
    "18 - 21 M3 (6 Ban Standard)",
    "22 - 33 M3 (6 Ban Long)",
    "34 - 40 M3 (Medium Truck)",
    "41 - 50 M3 (Medium Truck)",
    "51 - 60 M3 (Medium Truck Long)",
    ">60 M3 (Medium Truck Long)"
]

APLIKASI_OPTIONS = [
    "TRAILER",
    "NON-KUBIKASI (DUMP, MIXER, TANKI)",
    "BAK KAYU",
    "BAK BESI", 
    "BLIND VAN",
    "BOX ALUMINIUM",
    "BOX BESI",
    "DUMP TRUCK",
    "FLAT BED",
    "MEDIUM BUS",
    "MICROBUS",
    "MINI MIXER",
    "PICK UP",
    "WING BOX"
]

# =============================================================================
# API CLIENT FUNCTIONS
# =============================================================================

def check_server_health() -> bool:
    """Check if the Flask server is running and properly initialized"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"{Colors.OKGREEN}✅ Server Status: {data.get('message', 'Unknown')}{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}❌ Server returned error: {response.status_code}{Colors.ENDC}")
            if response.status_code == 500:
                try:
                    error_data = response.json()
                    print(f"{Colors.WARNING}⚠️  {error_data.get('message', 'Server error')}{Colors.ENDC}")
                except:
                    pass
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Colors.FAIL}❌ Cannot connect to server at {API_BASE_URL}{Colors.ENDC}")
        print(f"{Colors.WARNING}Make sure your Flask app is running: python app.py{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error checking server: {str(e)}{Colors.ENDC}")
        return False

def get_recommendations(criteria: Dict[str, str]) -> Optional[Dict]:
    """Get recommendations based on user criteria"""
    try:
        print(f"{Colors.WARNING}🔄 Sending criteria to AI system for analysis...{Colors.ENDC}")
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=criteria,
            headers={'Content-Type': 'application/json'},
            timeout=60  # Longer timeout for AI generation
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"{Colors.FAIL}❌ Recommendation error: {response.status_code}{Colors.ENDC}")
            try:
                error_data = response.json()
                print(f"{Colors.FAIL}Error details: {error_data.get('error', 'Unknown error')}{Colors.ENDC}")
            except:
                print(f"{Colors.FAIL}Response: {response.text}{Colors.ENDC}")
            return None
    except Exception as e:
        print(f"{Colors.FAIL}❌ Recommendation error: {str(e)}{Colors.ENDC}")
        return None

# =============================================================================
# USER INTERFACE FUNCTIONS
# =============================================================================

def print_header():
    """Print welcome header"""
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"🚗 CAR RECOMMENDATION SYSTEM - TERMINAL INTERFACE 🚗")
    print(f"{'='*60}{Colors.ENDC}")

def print_menu():
    """Print main menu options"""
    print(f"\n{Colors.OKBLUE}📋 MAIN MENU:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}1.{Colors.ENDC} Get vehicle recommendations")
    print(f"{Colors.OKCYAN}2.{Colors.ENDC} Exit")

def select_from_options(prompt: str, options: List[str]) -> str:
    """Generic function to select from a list of options"""
    while True:
        print(f"\n{Colors.OKBLUE}{prompt}{Colors.ENDC}")
        for i, option in enumerate(options, 1):
            print(f"{Colors.OKCYAN}{i:2d}.{Colors.ENDC} {option}")
        
        try:
            choice = int(input(f"\n{Colors.OKCYAN}Select option (1-{len(options)}): {Colors.ENDC}"))
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                print(f"{Colors.OKGREEN}✅ Selected: {selected}{Colors.ENDC}")
                return selected
            else:
                print(f"{Colors.WARNING}⚠️  Please select a number between 1 and {len(options)}{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.WARNING}⚠️  Please enter a valid number{Colors.ENDC}")

def get_criteria_input() -> Dict[str, str]:
    """Get all 5 criteria from user"""
    print(f"\n{Colors.HEADER}🎯 VEHICLE SELECTION CRITERIA{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Please select your requirements for each criterion:{Colors.ENDC}")
    
    criteria = {}
    
    # Get Segmentation
    criteria["segmentation"] = select_from_options(
        "🏢 INDUSTRY SEGMENTATION - What industry will this vehicle serve?",
        SEGMENTATION_OPTIONS
    )
    
    # Get Tipe Jalan
    criteria["tipe_jalan"] = select_from_options(
        "🛣️  ROAD TYPE - What type of roads will this vehicle primarily use?",
        TIPE_JALAN_OPTIONS
    )
    
    # Get Tonnase
    criteria["tonnase"] = select_from_options(
        "⚖️  WEIGHT CAPACITY - What tonnage capacity do you need?",
        TONNASE_OPTIONS
    )
    
    # Get Kubikasi Angkutan
    criteria["kubikasi_angkutan"] = select_from_options(
        "📦 VOLUME CAPACITY - What volume capacity do you need?",
        KUBIKASI_OPTIONS
    )
    
    # Get Aplikasi
    criteria["aplikasi"] = select_from_options(
        "🔧 APPLICATION TYPE - What will be the primary application?",
        APLIKASI_OPTIONS
    )
    
    return criteria

def display_criteria_summary(criteria: Dict[str, str]):
    """Display a summary of selected criteria"""
    print(f"\n{Colors.HEADER}📋 CRITERIA SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'─'*50}{Colors.ENDC}")
    print(f"{Colors.BOLD}🏢 Industry:{Colors.ENDC} {Colors.OKCYAN}{criteria['segmentation']}{Colors.ENDC}")
    print(f"{Colors.BOLD}🛣️  Road Type:{Colors.ENDC} {Colors.OKCYAN}{criteria['tipe_jalan']}{Colors.ENDC}")
    print(f"{Colors.BOLD}⚖️  Tonnage:{Colors.ENDC} {Colors.OKCYAN}{criteria['tonnase']}{Colors.ENDC}")
    print(f"{Colors.BOLD}📦 Volume:{Colors.ENDC} {Colors.OKCYAN}{criteria['kubikasi_angkutan']}{Colors.ENDC}")
    print(f"{Colors.BOLD}🔧 Application:{Colors.ENDC} {Colors.OKCYAN}{criteria['aplikasi']}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'─'*50}{Colors.ENDC}")

def display_structured_recommendations(response_data: Dict):
    """Display the structured JSON response in a nice format"""
    if not response_data or 'recommendations' not in response_data:
        print(f"{Colors.FAIL}❌ Invalid response format{Colors.ENDC}")
        return
    
    recommendations = response_data['recommendations']
    
    print(f"\n{Colors.OKGREEN}✅ AI RECOMMENDATIONS GENERATED{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}🎯 System found {len(recommendations)} matching vehicles for your criteria{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    # Display each recommendation
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{Colors.HEADER}🚗 RECOMMENDATION #{i}")
        print(f"{'─'*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}Vehicle:{Colors.ENDC} {Colors.OKBLUE}{rec.get('product_name', 'Unknown')}{Colors.ENDC}")
        print(f"{Colors.BOLD}Label:{Colors.ENDC} {Colors.OKGREEN}💡 {rec.get('label', 'Great Choice')}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}📝 Why This Vehicle Suits Your Needs:{Colors.ENDC}")
        reason = rec.get('reason', 'This vehicle offers excellent value and performance.')
        # Word wrap the reason for better readability
        words = reason.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 65:  # 65 chars per line
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            print(f"{Colors.OKCYAN}{line}{Colors.ENDC}")
        
        if i < len(recommendations):
            print(f"\n{Colors.HEADER}{'─'*60}{Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}🎉 Analysis complete! These are your top vehicle matches.{Colors.ENDC}")

def display_raw_json(response_data: Dict):
    """Display the raw JSON response for debugging"""
    print(f"\n{Colors.HEADER}📋 RAW JSON RESPONSE:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{json.dumps(response_data, indent=2)}{Colors.ENDC}")

def handle_get_recommendation():
    """Handle getting recommendation with criteria input"""
    print(f"\n{Colors.OKBLUE}🎯 GET VEHICLE RECOMMENDATIONS{Colors.ENDC}")
    print(f"{Colors.OKCYAN}We'll help you find the perfect vehicle by analyzing your specific needs.{Colors.ENDC}")
    
    # Get criteria from user
    criteria = get_criteria_input()
    
    # Display summary
    display_criteria_summary(criteria)
    
    # Confirm with user
    confirm = input(f"\n{Colors.OKCYAN}Proceed with these criteria? (y/n): {Colors.ENDC}").strip().lower()
    if confirm not in ['y', 'yes']:
        print(f"{Colors.WARNING}📝 Cancelled. Returning to main menu.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.WARNING}🔄 Analyzing your criteria and finding matching vehicles...{Colors.ENDC}")
    print(f"{Colors.OKCYAN}This may take a moment as we:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  1. Match your criteria against our vehicle database{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  2. Calculate compatibility scores{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  3. Select the top 3 matching vehicles{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  4. Generate AI-powered recommendations{Colors.ENDC}")
    
    response_data = get_recommendations(criteria)
    
    if not response_data:
        print(f"{Colors.FAIL}❌ Could not generate recommendations{Colors.ENDC}")
        return
    
    # Display structured recommendations
    display_structured_recommendations(response_data)
    
    # Ask if user wants to see raw JSON
    show_json = input(f"\n{Colors.OKCYAN}Show technical details (raw JSON)? (y/n): {Colors.ENDC}").strip().lower()
    if show_json in ['y', 'yes']:
        display_raw_json(response_data)

def main():
    """Main application loop"""
    print_header()
    print(f"{Colors.OKCYAN}Welcome! This system will analyze your specific vehicle requirements")
    print(f"and recommend the top 3 matching Isuzu Commercial Vehicles.{Colors.ENDC}")
    
    # Check server connection
    print(f"\n{Colors.WARNING}🔄 Checking server connection...{Colors.ENDC}")
    if not check_server_health():
        print(f"\n{Colors.FAIL}Cannot continue without server connection. Exiting.{Colors.ENDC}")
        sys.exit(1)
    
    while True:
        try:
            print_menu()
            choice = input(f"\n{Colors.OKCYAN}Select an option (1-2): {Colors.ENDC}")
            
            if choice == '1':
                handle_get_recommendation()
            elif choice == '2':
                print(f"\n{Colors.OKGREEN}👋 Thanks for using Car Recommendation System!{Colors.ENDC}")
                break
            else:
                print(f"{Colors.WARNING}⚠️  Please select a valid option (1-2){Colors.ENDC}")
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
            break
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
            print(f"{Colors.WARNING}Continuing...{Colors.ENDC}")

if __name__ == "__main__":
    # Check if requests module is available
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' module not found!")
        print("Install it with: pip install requests")
        sys.exit(1)
    
    main()

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
"""
HOW TO USE THIS ENHANCED TERMINAL INTERFACE:

1. FIRST: Make sure your Flask app is running
   python app.py

2. THEN: Run this terminal interface in another terminal
   python term.py

3. FEATURES:
   - Option 1: Get AI recommendations (requires exactly 3 products)
   - Option 2: Exit

4. REQUIREMENTS:
   pip install requests

5. STRUCTURED WORKFLOW (API requires exactly 3 products):
   - Add Product Name and Score (repeat 3 times)
   - Provide context for recommendations
   - Get AI-powered structured recommendations
   - View beautifully formatted output
   - Optional: View raw JSON response

6. ENHANCED OPTIONS AFTER ADDING EACH PRODUCT:
   - Add another product (up to 3 max)
   - Get recommendations (only when you have exactly 3)
   - Remove a product from the list
   - Change context
   - Clear all and start over
   - Cancel and return to main menu

EXAMPLE WORKFLOW:
1. Select option 1 (Get recommendations)
2. Enter context: "The user is working in retail"
3. Enter Product Name: "Toyota Camry", Score: 85
4. Enter Product Name: "Honda Accord", Score: 78  
5. Enter Product Name: "Tesla Model 3", Score: 92
6. Choose option 2 to get AI recommendations
7. View beautifully formatted recommendations!

The system will generate personalized recommendations for each product with:
- Product name and score
- AI-generated label (2-4 words)
- Detailed reason (80-120 words)
- Word-wrapped formatting for readability
- Optional raw JSON view for debugging

API COMPATIBILITY:
- Matches app.py's Pydantic models exactly
- Validates input/output structure
- Handles errors gracefully
- Provides clear feedback on requirements
"""