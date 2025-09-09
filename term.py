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

# We don't need these functions anymore for the simplified version
def get_all_cars() -> List[str]:
    """Get list of all available cars - kept for potential future use"""
    try:
        response = requests.get(f"{API_BASE_URL}/cars", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('cars', [])
        else:
            print(f"{Colors.FAIL}❌ Error getting car list: {response.status_code}{Colors.ENDC}")
            return []
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {str(e)}{Colors.ENDC}")
        return []

def search_cars(query: str, limit: int = 5) -> List[Dict]:
    """Search for cars using AI similarity - kept for potential future use"""
    try:
        payload = {
            "query": query,
            "limit": limit
        }
        response = requests.post(
            f"{API_BASE_URL}/cars/search", 
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            print(f"{Colors.FAIL}❌ Search error: {response.status_code}{Colors.ENDC}")
            try:
                error_data = response.json()
                print(f"{Colors.FAIL}Error details: {error_data.get('error', 'Unknown error')}{Colors.ENDC}")
            except:
                print(f"{Colors.FAIL}Response: {response.text}{Colors.ENDC}")
            return []
    except Exception as e:
        print(f"{Colors.FAIL}❌ Search error: {str(e)}{Colors.ENDC}")
        return []

def get_recommendations(products: List[Dict], context: str) -> Optional[Dict]:
    """Get detailed recommendations for selected cars with individual scores"""
    try:
        # Validate input - API requires exactly 3 recommendations
        if len(products) != 3:
            print(f"{Colors.WARNING}⚠️  API requires exactly 3 products. You have {len(products)}.{Colors.ENDC}")
            return None
        
        # Create recommendations list with product_name and score for each product
        recommendations = []
        for product in products:
            recommendations.append({
                "product_name": product['name'],
                "score": str(product['score'])  # Convert to string as per API requirement
            })
        
        payload = {
            "context": context,
            "recommendation": recommendations
        }
        
        print(f"{Colors.WARNING}🔄 Sending request to AI system...{Colors.ENDC}")
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
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
    print(f"{Colors.OKCYAN}1.{Colors.ENDC} Get recommendations")
    print(f"{Colors.OKCYAN}2.{Colors.ENDC} Exit")

def get_product_input() -> Dict:
    """Get a single product name and score from user"""
    # Get Product Name
    while True:
        product_name = input(f"{Colors.OKCYAN}Enter Product Name: {Colors.ENDC}").strip()
        if product_name:
            break
        print(f"{Colors.WARNING}⚠️  Please enter a product name{Colors.ENDC}")
    
    # Get Score
    while True:
        try:
            score = int(input(f"{Colors.OKCYAN}Enter Score (0-100): {Colors.ENDC}"))
            if 0 <= score <= 100:
                break
            print(f"{Colors.WARNING}⚠️  Score must be between 0 and 100{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.WARNING}⚠️  Please enter a valid number for score{Colors.ENDC}")
    
    return {"name": product_name, "score": score}

def display_current_products(products: List[Dict]):
    """Display the current list of products with validation status"""
    if not products:
        print(f"{Colors.WARNING}📝 No products added yet{Colors.ENDC}")
        return
    
    print(f"\n{Colors.OKBLUE}📝 Current Products ({len(products)}/3 required):{Colors.ENDC}")
    for i, product in enumerate(products, 1):
        print(f"{Colors.OKCYAN}{i}.{Colors.ENDC} {product['name']} (Score: {product['score']}/100)")
    
    if len(products) < 3:
        print(f"{Colors.WARNING}⚠️  Need {3 - len(products)} more product(s) for recommendations{Colors.ENDC}")
    elif len(products) == 3:
        print(f"{Colors.OKGREEN}✅ Ready for recommendations!{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}⚠️  Too many products! API accepts exactly 3.{Colors.ENDC}")

def get_context_input() -> str:
    """Get context from user"""
    print(f"\n{Colors.OKBLUE}📝 CONTEXT SETUP{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Please provide context for the recommendations.{Colors.ENDC}")
    print(f"{Colors.WARNING}Examples:{Colors.ENDC}")
    print(f"  • 'The user is working in retail'")
    print(f"  • 'Customer is a young professional'")
    print(f"  • 'Family with two kids looking for safety'")
    print(f"  • 'Budget-conscious first-time buyer'")
    
    while True:
        context = input(f"\n{Colors.OKCYAN}Enter context: {Colors.ENDC}").strip()
        if context:
            return context
        print(f"{Colors.WARNING}⚠️  Please enter a context{Colors.ENDC}")

def display_structured_recommendations(response_data: Dict, products: List[Dict], context: str):
    """Display the structured JSON response in a nice format"""
    if not response_data or 'recommendations' not in response_data:
        print(f"{Colors.FAIL}❌ Invalid response format{Colors.ENDC}")
        return
    
    recommendations = response_data['recommendations']
    
    print(f"\n{Colors.OKGREEN}✅ AI RECOMMENDATIONS GENERATED{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Context: '{context}'{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    # Display each recommendation
    for i, rec in enumerate(recommendations, 1):
        # Find the corresponding score from products list
        product_score = next((p['score'] for p in products if p['name'] == rec.get('product_name', 'Unknown')), 'N/A')
        
        print(f"\n{Colors.HEADER}🚗 RECOMMENDATION #{i}")
        print(f"{'─'*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}Product:{Colors.ENDC} {Colors.OKBLUE}{rec.get('product_name', 'Unknown')}{Colors.ENDC}")
        print(f"{Colors.BOLD}Score:{Colors.ENDC} {Colors.OKCYAN}{product_score}/100{Colors.ENDC}")
        print(f"{Colors.BOLD}Label:{Colors.ENDC} {Colors.OKGREEN}💡 {rec.get('label', 'Great Choice')}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}📝 Detailed Recommendation:{Colors.ENDC}")
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
    print(f"{Colors.OKGREEN}🎉 All recommendations generated successfully!{Colors.ENDC}")

def display_raw_json(response_data: Dict):
    """Display the raw JSON response for debugging"""
    print(f"\n{Colors.HEADER}📋 RAW JSON RESPONSE:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{json.dumps(response_data, indent=2)}{Colors.ENDC}")

def handle_get_recommendation():
    """Handle getting recommendation with context and multiple products"""
    print(f"\n{Colors.OKBLUE}🎯 GET RECOMMENDATIONS{Colors.ENDC}")
    
    # First, get context from user
    context = get_context_input()
    print(f"{Colors.OKGREEN}✅ Context set: '{context}'{Colors.ENDC}")
    
    products = []
    
    while True:
        print(f"\n{Colors.HEADER}{'─'*40}")
        print(f"Adding Product #{len(products) + 1}")
        print(f"{'─'*40}{Colors.ENDC}")
        
        # Get product input
        product = get_product_input()
        products.append(product)
        
        # Display current products
        display_current_products(products)
        
        # Ask if user wants to add more products
        print(f"\n{Colors.OKBLUE}Options:{Colors.ENDC}")
        if len(products) < 3:
            print(f"{Colors.OKCYAN}1.{Colors.ENDC} Add another product")
        else:
            print(f"{Colors.WARNING}1.{Colors.ENDC} Add another product (max 3 reached)")
        
        if len(products) == 3:
            print(f"{Colors.OKGREEN}2.{Colors.ENDC} Get AI recommendations (ready!)")
        else:
            print(f"{Colors.WARNING}2.{Colors.ENDC} Get AI recommendations (need {3-len(products)} more)")
        
        print(f"{Colors.OKCYAN}3.{Colors.ENDC} Remove a product")
        print(f"{Colors.OKCYAN}4.{Colors.ENDC} Change context")
        print(f"{Colors.OKCYAN}5.{Colors.ENDC} Clear all and start over")
        print(f"{Colors.OKCYAN}6.{Colors.ENDC} Cancel and return to main menu")
        
        while True:
            choice = input(f"{Colors.OKCYAN}Select option (1-6): {Colors.ENDC}").strip()
            
            if choice == '1':
                # Continue loop to add another product
                if len(products) >= 3:
                    print(f"{Colors.WARNING}⚠️  Maximum 3 products allowed for API compatibility{Colors.ENDC}")
                    continue
                break
            elif choice == '2':
                # Get recommendations
                if len(products) != 3:
                    print(f"{Colors.WARNING}⚠️  Need exactly 3 products for recommendations. You have {len(products)}.{Colors.ENDC}")
                    continue
                
                print(f"\n{Colors.WARNING}🔄 Generating AI recommendations for {len(products)} products...{Colors.ENDC}")
                print(f"{Colors.OKBLUE}Context: '{context}'{Colors.ENDC}")
                
                response_data = get_recommendations(products, context)
                
                if not response_data:
                    print(f"{Colors.FAIL}❌ Could not generate recommendations{Colors.ENDC}")
                    return
                
                # Display structured recommendations
                display_structured_recommendations(response_data, products, context)
                
                # Ask if user wants to see raw JSON
                show_json = input(f"\n{Colors.OKCYAN}Show raw JSON response? (y/n): {Colors.ENDC}").strip().lower()
                if show_json in ['y', 'yes']:
                    display_raw_json(response_data)
                
                return
            elif choice == '3':
                # Remove a product
                if len(products) <= 1:
                    print(f"{Colors.WARNING}⚠️  Cannot remove the only product. Use option 5 to start over.{Colors.ENDC}")
                    continue
                
                display_current_products(products)
                try:
                    remove_idx = int(input(f"{Colors.OKCYAN}Enter number of product to remove: {Colors.ENDC}")) - 1
                    if 0 <= remove_idx < len(products):
                        removed = products.pop(remove_idx)
                        print(f"{Colors.OKGREEN}✅ Removed: {removed['name']}{Colors.ENDC}")
                    else:
                        print(f"{Colors.WARNING}⚠️  Invalid product number{Colors.ENDC}")
                except ValueError:
                    print(f"{Colors.WARNING}⚠️  Please enter a valid number{Colors.ENDC}")
                break
            elif choice == '4':
                # Change context
                context = get_context_input()
                print(f"{Colors.OKGREEN}✅ Context updated: '{context}'{Colors.ENDC}")
                break
            elif choice == '5':
                # Clear all and start over
                products = []
                context = get_context_input()
                print(f"{Colors.OKGREEN}✅ Cleared all products and reset context{Colors.ENDC}")
                break
            elif choice == '6':
                # Cancel
                print(f"{Colors.WARNING}📝 Cancelled. Returning to main menu.{Colors.ENDC}")
                return
            else:
                print(f"{Colors.WARNING}⚠️  Please select a valid option (1-6){Colors.ENDC}")

def main():
    """Main application loop"""
    print_header()
    
    # Check server connection
    print(f"{Colors.WARNING}🔄 Checking server connection...{Colors.ENDC}")
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