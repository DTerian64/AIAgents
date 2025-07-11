"""
Cosmicworks AI Tool - Product Data Aggregation Tool
This tool aggregates and processes product data from CosmosDB
"""

from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProductSummary:
    """Data class for aggregated product information"""
    name: str
    description: str
    category: str
    sub_category: str
    sku: str
    tags: List[str]
    cost: float
    price: float
    type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "sub_category": self.sub_category,
            "sku": self.sku,
            "tags": self.tags,
            "cost": self.cost,
            "price": self.price,
            "type": self.type
        }
    
    def __str__(self) -> str:
        """String representation for display"""
        return f"{self.name} ({self.sku}) - {self.category}/{self.sub_category} - ${self.price}"


class CosmicworksAITool:
    """AI Tool for processing and aggregating Cosmicworks product data"""
    
    def __init__(self):
        self.products: List[ProductSummary] = []
    
    def extract_product_fields(self, product_data: Dict[str, Any]) -> ProductSummary:
        """
        Extract and aggregate specific fields from a product document
        
        Args:
            product_data: Raw product document from CosmosDB
            
        Returns:
            ProductSummary object with aggregated fields
        """
        try:
            # Extract category and subcategory
            category = product_data.get("category", {})
            category_name = category.get("name", "Unknown") if isinstance(category, dict) else str(category)
            
            sub_category = category.get("subCategory", {}) if isinstance(category, dict) else {}
            sub_category_name = sub_category.get("name", "Unknown") if isinstance(sub_category, dict) else str(sub_category)
            
            # Create ProductSummary object
            product_summary = ProductSummary(
                name=product_data.get("name", "Unknown"),
                description=product_data.get("description", ""),
                category=category_name,
                sub_category=sub_category_name,
                sku=product_data.get("sku", ""),
                tags=product_data.get("tags", []),
                cost=float(product_data.get("cost", 0.0)),
                price=float(product_data.get("price", 0.0)),
                type=product_data.get("type", "Unknown")
            )
            
            return product_summary
            
        except Exception as e:
            print(f"Error processing product data: {e}")
            # Return a default ProductSummary with available data
            return ProductSummary(
                name=product_data.get("name", "Unknown"),
                description="Error processing product",
                category="Unknown",
                sub_category="Unknown",
                sku=product_data.get("sku", ""),
                tags=[],
                cost=0.0,
                price=0.0,
                type="Unknown"
            )
    
    def aggregate_products(self, products_data: List[Dict[str, Any]]) -> bool:
        """
        Aggregate multiple product documents
        
        Args:
            products_data: List of product documents from CosmosDB
            
        Returns:
            bool: True if aggregation was successful, False otherwise
        """
        try:
            self.products = []
            
            for product_data in products_data:
                product_summary = self.extract_product_fields(product_data)
                self.products.append(product_summary)
            
            return True
        except Exception as e:
            print(f"Error aggregating products: {e}")
            return False
    
    def get_products(self) -> List[ProductSummary]:
        """
        Get the list of aggregated products
        
        Returns:
            List of ProductSummary objects
        """
        return self.products
        """Filter products by category"""
        return [p for p in self.products if p.category.lower() == category.lower()]
    
    def filter_by_category(self, category: str) -> List[ProductSummary]:
        """Filter products by subcategory"""
        return [p for p in self.products if p.sub_category.lower() == subcategory.lower()]
    
    def filter_by_price_range(self, min_price: float, max_price: float) -> List[ProductSummary]:
        """Filter products by price range"""
        return [p for p in self.products if min_price <= p.price <= max_price]
    
    def filter_by_tags(self, tag: str) -> List[ProductSummary]:
        """Filter products by tag"""
        return [p for p in self.products if tag.lower() in [t.lower() for t in p.tags]]
    
    def search_products(self, query: str) -> List[ProductSummary]:
        """
        Search products by name, description, or SKU
        
        Args:
            query: Search query string
            
        Returns:
            List of matching ProductSummary objects
        """
        query_lower = query.lower()
        results = []
        
        for product in self.products:
            if (query_lower in product.name.lower() or 
                query_lower in product.description.lower() or 
                query_lower in product.sku.lower()):
                results.append(product)
        
        return results
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get summary of products by category"""
        category_counts = {}
        for product in self.products:
            category = product.category
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def get_price_statistics(self) -> Dict[str, float]:
        """Get price statistics for all products"""
        if not self.products:
            return {}
        
        prices = [p.price for p in self.products]
        costs = [p.cost for p in self.products]
        
        return {
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": sum(prices) / len(prices),
            "min_cost": min(costs),
            "max_cost": max(costs),
            "avg_cost": sum(costs) / len(costs),
            "avg_margin": sum([(p.price - p.cost) for p in self.products]) / len(self.products)
        }
    
    def export_to_json(self, filename: str = None) -> str:
        """
        Export aggregated products to JSON
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of aggregated products
        """
        products_dict = [product.to_dict() for product in self.products]
        json_data = json.dumps(products_dict, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_data)
            print(f"Products exported to {filename}")
        
        return json_data
    
    def generate_ai_context(self, max_products: int = 10) -> str:
        """
        Generate AI-friendly context string for the aggregated products
        
        Args:
            max_products: Maximum number of products to include in context
            
        Returns:
            Formatted string suitable for AI model context
        """
        if not self.products:
            return "No products available."
        
        # Get category summary
        category_summary = self.get_category_summary()
        price_stats = self.get_price_statistics()
        
        context = f"Product Database Summary:\n"
        context += f"- Total Products: {len(self.products)}\n"
        context += f"- Categories: {', '.join(category_summary.keys())}\n"
        context += f"- Price Range: ${price_stats.get('min_price', 0):.2f} - ${price_stats.get('max_price', 0):.2f}\n\n"
        
        # Add sample products
        context += f"Sample Products (showing first {min(max_products, len(self.products))}):\n"
        for i, product in enumerate(self.products[:max_products]):
            context += f"{i+1}. {product.name} ({product.sku})\n"
            context += f"   Category: {product.category} > {product.sub_category}\n"
            context += f"   Price: ${product.price:.2f}\n"
            context += f"   Description: {product.description[:100]}...\n\n"
        
        return context


# Example usage and helper functions
def process_cosmicworks_data(products_data: List[Dict[str, Any]]) -> CosmicworksAITool:
    """
    Helper function to process CosmicWorks product data
    
    Args:
        products_data: List of product documents from CosmosDB
        
    Returns:
        CosmicworksAITool instance with processed data
    """
    tool = CosmicworksAITool()
    tool.aggregate_products(products_data)
    return tool


# Example usage
if __name__ == "__main__":
    # Example product data (as provided)
    sample_product = {
        "id": "00000000-0000-0000-0000-000000005020",
        "name": "ML Fork",
        "description": "Composite road fork with an aluminum steerer tube.",
        "category": {
            "name": "Components",
            "subCategory": {
                "name": "Forks"
            }
        },
        "sku": "FK-5136",
        "tags": ["Components", "Forks"],
        "cost": 77.9176,
        "price": 175.49,
        "type": "Product"
    }
    
    # Initialize tool and process data
    tool = CosmicworksAITool()
    products = [sample_product]  # In real use, this would be your full product list
    
    # Aggregate the data
    success = tool.aggregate_products(products)
    
    if success:
        # Display results
        print("Aggregated Products:")
        for product in tool.get_products():
            print(product)
        
        # Generate AI context
        print("\nAI Context:")
        print(tool.generate_ai_context())
        
        # Export to JSON
        json_output = tool.export_to_json()
        print("\nJSON Export:")
        print(json_output)
    else:
        print("Failed to aggregate products")