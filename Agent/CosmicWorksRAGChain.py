from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
import spacy
import os
from typing import Optional, List, Any
import numpy as np
import hashlib
# import Helpers.MyCosmosDBHelper as MyCosmosDBHelper


class AzureEmbeddings(Embeddings):
    """Custom embeddings using Azure ChatCompletionsClient"""
    
    def __init__(self, client, model_name: str = "text-embedding-ada-002"):
        self.client = client
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Azure"""
        embeddings = []
        for text in texts:
            embedding = self._create_simple_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Azure"""
        return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple embedding - replace with actual Azure embedding call"""
        # This is a placeholder - you should replace this with actual Azure embedding API call
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to numbers and normalize
        numbers = [int(hash_hex[i:i+2], 16) for i in range(0, min(len(hash_hex), 32), 2)]
        while len(numbers) < 1536:  # Standard embedding size
            numbers.extend(numbers[:1536-len(numbers)])
        numbers = numbers[:1536]
        
        # Normalize to [-1, 1]
        embedding = [(n - 127.5) / 127.5 for n in numbers]
        return embedding


class AzureChatCompletionsLLM(LLM):
    """Custom LangChain LLM wrapper for Azure ChatCompletionsClient"""
    
    def __init__(self, client, model_name: str):
        super().__init__()
        self.client = client
        self.model_name = model_name
    
    @property
    def _llm_type(self) -> str:
        return "azure_chat_completions"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Azure ChatCompletionsClient for Ministral-3B"""
        try:
            # Format messages for Azure Foundry
            messages = [{"role": "user", "content": prompt}]
            
            # Call the Azure client - adjust parameters for Ministral-3B
            response = self.client.complete(
                messages=messages,
                model=self.model_name,  # Should be "Ministral-3B" or similar
                max_tokens=kwargs.get("max_tokens", 1024),  # Reduced for smaller model
                temperature=kwargs.get("temperature", 0.7),  # Slightly lower for better consistency
                top_p=kwargs.get("top_p", 0.9),  # Adjusted for better performance
                stream=False  # Ensure non-streaming response
            )
            
            # Extract the response content
            return response.choices[0].message.content
            
        except Exception as e:
            # Enhanced error handling with more specific error info
            error_msg = f"Azure Foundry API call failed: {str(e)}"
            
            # Check if it's a 404 error specifically
            if "404" in str(e):
                error_msg += f"\nPossible issues:\n- Model name '{self.model_name}' not found\n- Endpoint URL incorrect\n- Model not deployed or accessible"
            elif "401" in str(e) or "403" in str(e):
                error_msg += f"\nAuthentication issue - check your API key"
            
            raise Exception(error_msg)


class CosmicWorksRAGChain:
    def __init__(self, client=None, model_name="Ministral-3B", embedding_model="text-embedding-ada-002", 
                 azure_endpoint=None, azure_api_key=None, use_openai_fallback=False):
        self.model_name = model_name
        self.use_openai_fallback = use_openai_fallback
        
        # Load spaCy model with error handling
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize LLM and embeddings
        self._initialize_llm_and_embeddings(client, model_name, embedding_model, azure_endpoint, azure_api_key)
        
        self.vectorstore = None
    
    def _initialize_llm_and_embeddings(self, client, model_name, embedding_model, azure_endpoint, azure_api_key):
        """Initialize LLM and embeddings with proper error handling for Azure Foundry"""
        if client:
            try:
                print(f"Initializing Azure Foundry with model: {model_name}")
                self.llm = AzureChatCompletionsLLM(client, model_name)
                
                # For Azure Foundry, we'll use custom embeddings since Azure OpenAI might not be available
                # You can modify this if you have access to Azure OpenAI embeddings
                if azure_endpoint and azure_api_key:
                    try:
                        self.embeddings = AzureOpenAIEmbeddings(
                            azure_endpoint=azure_endpoint,
                            azure_deployment=embedding_model,
                            api_key=azure_api_key,
                            api_version="2024-05-01-preview"
                        )
                        print("Using Azure OpenAI embeddings")
                    except Exception as e:
                        print(f"Azure OpenAI embeddings failed: {e}, falling back to custom embeddings")
                        self.embeddings = AzureEmbeddings(client, embedding_model)
                else:
                    # Use custom Azure embeddings with the same client
                    self.embeddings = AzureEmbeddings(client, embedding_model)
                    print("Using custom Azure embeddings")
                    
            except Exception as e:
                print(f"Warning: Azure Foundry initialization failed: {e}")
                if self.use_openai_fallback:
                    self._fallback_to_openai("gpt-3.5-turbo")  # Use OpenAI model name for fallback
                else:
                    raise e
        else:
            self._fallback_to_openai("gpt-3.5-turbo")
    
    def _fallback_to_openai(self, model_name):
        """Fallback to OpenAI if Azure fails"""
        try:
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.8)
            print("Using OpenAI fallback")
        except Exception as e:
            raise Exception(f"Both Azure and OpenAI initialization failed: {e}")
    
    def parse_natural_query(self, user_input):
        """Enhanced spaCy parsing with better keyword extraction"""
        if not self.nlp:
            # Simple fallback parsing if spaCy is not available
            words = user_input.lower().split()
            keywords = [word for word in words if word not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
            return keywords, user_input, [], keywords[0] if keywords else None, []
        
        doc = self.nlp(user_input)
        
        # Extract entities and keywords
        entities = [ent.text for ent in doc.ents]
        
        # Better keyword extraction - prioritize nouns and product-related terms
        keywords = []
        product_terms = []
        qualifiers = []
        
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
                
            # Prioritize nouns (likely product names/categories)
            if token.pos_ in ['NOUN', 'PROPN']:
                product_terms.append(token.lemma_.lower())
            # Capture adjectives that might be relevant (price, quality, etc.)
            elif token.pos_ in ['ADJ']:
                qualifiers.append(token.lemma_.lower())
            # Other meaningful words
            elif token.pos_ not in ['DET', 'ADP', 'CONJ', 'PRON']:
                keywords.append(token.lemma_.lower())
        
        # Combine all keywords but prioritize product terms
        all_keywords = product_terms + keywords + qualifiers
        
        # Enhanced query processing - preserve important context
        processed_query = " ".join([token.text for token in doc if not token.is_stop])
        
        # Try to identify the main product category
        main_product = None
        if product_terms:
            main_product = product_terms[0]  # Use first noun as main product
        elif keywords:
            main_product = keywords[0]
        
        return all_keywords, processed_query, entities, main_product, qualifiers
    
    def create_vectorstore_from_products(self, products):
        """Convert Cosmos DB results to LangChain documents and create vectorstore"""
        documents = []
        
        for product in products:
            # Create document content from product data
            content = f"""
            Product: {product.get('name', 'Unknown')}
            Category: {product.get('category', 'Unknown')}
            Description: {product.get('description', 'No description')}
            Price: ${product.get('price', 0)}
            Tags: {', '.join(product.get('tags', []))}
            """
            
            # Create LangChain Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'product_id': product.get('id'),
                    'category': product.get('category'),
                    'price': product.get('price'),
                    'name': product.get('name')
                }
            )
            documents.append(doc)
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        return self.vectorstore
    
    def create_rag_chain(self):
        """Create a RAG chain using LangChain"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore_from_products first.")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create prompt template optimized for Ministral-3B
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant for Cosmic Works product database.
        
        Context: {context}
        
        Question: {input}
        
        Instructions:
        - Find the cheapest product if user asks for "cheapest", "most affordable", or "lowest price"
        - Find the most expensive product if user asks for "most expensive", "premium", or "highest price"
        - Include product name, price, and key details in your response
        - Be concise and direct
        
        Answer:
        """)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    def get_process_cosmic_langchain(self, user_input):
        """Main function using LangChain RAG with improved search logic"""
        try:
            # Parse input with spaCy
            all_keywords, processed_query, entities, main_product, qualifiers = self.parse_natural_query(user_input)
            
            # Smart search strategy
            results = []
            
            # Strategy 1: Use main product (noun) for search
            if main_product:
                results = MyCosmosDBHelper.search_products_detailed(main_product)
                
            # Strategy 2: If no results, try other keywords
            if not results and all_keywords:
                for keyword in all_keywords:
                    results = MyCosmosDBHelper.search_products_detailed(keyword)
                    if results:
                        break
            
            # Strategy 3: If still no results, try broader search or category search
            if not results:
                # You might want to implement a category-based search here
                # For now, let's try the first entity or processed query
                if entities:
                    results = MyCosmosDBHelper.search_products_detailed(entities[0])
                elif processed_query:
                    # Try searching with the processed query
                    results = MyCosmosDBHelper.search_products_detailed(processed_query.split()[0])
            
            if not results:
                return f"No matching products found for your query: '{user_input}'. Try searching for specific product names or categories."
            
            # Create vectorstore from results
            self.create_vectorstore_from_products(results)
            
            # Create RAG chain
            rag_chain = self.create_rag_chain()
            
            # Enhanced prompt that includes the qualifiers (like "cheapest")
            enhanced_query = user_input
            if qualifiers:
                qualifier_context = f"The user is looking for products that are: {', '.join(qualifiers)}. "
                enhanced_query = qualifier_context + user_input
            
            # Get response using RAG
            response = rag_chain.invoke({"input": enhanced_query})
            
            return response["answer"]
            
        except Exception as e:
            return f"Error processing your request: {str(e)}"


# Example usage and testing
def main():
    """Example usage with Azure Foundry and Ministral-3B"""
    
    try:
        # Azure Foundry setup with Ministral-3B
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
            
            # Your Azure Foundry endpoint and credentials
            endpoint = "your-azure-foundry-endpoint"  # e.g., https://your-resource.inference.ai.azure.com
            api_key = os.getenv("AZURE_KEY")
            
            client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
                api_version="2024-05-01-preview"
            )
            
            # Initialize with Ministral-3B
            cosmic_rag = CosmicWorksRAGChain(
                client=client,
                model_name="Ministral-3B",  # Exact model name as deployed
                use_openai_fallback=False  # Don't fallback to OpenAI
            )
            
            print("Successfully initialized Azure Foundry with Ministral-3B")
            
        except Exception as azure_error:
            print(f"Azure Foundry setup failed: {azure_error}")
            raise azure_error
        
        # Test the system
        user_query = "Show me the cheapest bike"
        print(f"Processing query: {user_query}")
        response = cosmic_rag.get_process_cosmic_langchain(user_query)
        print("Response:", response)
        
    except Exception as e:
        print(f"Error in main: {e}")
        
        # Debug information
        print("\nTroubleshooting steps:")
        print("1. Verify your Azure Foundry endpoint URL")
        print("2. Check that Ministral-3B is deployed and accessible")
        print("3. Confirm your API key is correct")
        print("4. Ensure the model name matches exactly as deployed")
        

if __name__ == "__main__":
    main()