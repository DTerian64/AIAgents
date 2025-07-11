import os
from azure.cosmos import CosmosClient, PartitionKey

from dotenv import load_dotenv
load_dotenv()

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT") 
COSMOS_KEY = os.getenv("COSMOS_PRIMARY_KEY")
AICCONVERSATIONS_DATABASE_NAME = os.getenv("AICCONVERSATIONS_DATABASE_NAME") 
COSMICWORKS_DATABASE_NAME = os.getenv("COSMICWORKS_DATABASE_NAME")

cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)


def getAIConversationsContainer():    
    return cosmos_client.get_database_client(AICCONVERSATIONS_DATABASE_NAME).get_container_client("AIConversations")

def getProductsContainer():
    print("Connecting to Products container in CosmicWorks database")
    return cosmos_client.get_database_client(COSMICWORKS_DATABASE_NAME).get_container_client("products")
def getEmployeeContainer():
    return cosmos_client.get_database_client(COSMICWORKS_DATABASE_NAME).get_container_client("employees")

def search_products(keyword):
    print(f"Searching for products with keyword: {keyword}")
    query = f"SELECT * FROM products p WHERE CONTAINS(LOWER(p.name), LOWER(@keyword))"
    items = list(getProductsContainer().query_items(
        query=query,
        parameters=[{"name": "@keyword", "value": keyword}],
        enable_cross_partition_query=True
    ))
    return items

def search_products_detailed(keyword: str):    
    query = """
    SELECT p.id, p.name, p.description, p.category.name AS category,
           p.category.subCategory.name AS subCategory,
           p.sku, p.tags, p.cost, p.price, p.type
    FROM products p
    WHERE 
        CONTAINS(LOWER(p.name), LOWER(@keyword)) OR
        CONTAINS(LOWER(p.description), LOWER(@keyword)) OR
        CONTAINS(LOWER(p.category.name), LOWER(@keyword)) OR
        CONTAINS(LOWER(p.category.subCategory.name), LOWER(@keyword)) OR
        CONTAINS(LOWER(p.sku), LOWER(@keyword)) OR
        ARRAY_CONTAINS(p.tags, @keyword, true)
    """
    items = list(getProductsContainer().query_items(
        query=query,
        parameters=[{"name": "@keyword", "value": keyword}],
        enable_cross_partition_query=True
    ))

    return items

def search_employees(keyword):
    query = f"SELECT * FROM employees e WHERE CONTAINS(LOWER(e.name), LOWER(@keyword))"
    items = list(getEmployeeContainer().query_items(
        query=query,
        parameters=[{"name": "@keyword", "value": keyword}],
        enable_cross_partition_query=True
    ))
    return items