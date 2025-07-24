## Application Workflow

1. **User Login**
   - Uses MSAL JavaScript to authenticate against Azure Entra ID.
   - Stores token in sessionStorage.

2. **Question Submission**
   - User submits question from frontend.
   - JavaScript sends POST request to `/api/chat` with token and question.

3. **Token Verification**
   - FastAPI validates JWT using Microsoft JWKS endpoint.

4. **AI Routing**
   - `David64OpenAI.get_agent_response()` decides between general or LangChain flow.

5. **LangChain Flow (CosmicWorksLangChain)**
   - Loads FAISS index from blob storage if not available locally.
   - Retrieves relevant documents.
   - Returns an answer using a prompt template.

6. **Logging to CosmosDB**
   - The question and answer are saved for traceability.

7. **Response to Frontend**
   - JSON response returned and displayed in textarea.