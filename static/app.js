const msalConfig = {
  auth: {
    clientId: "81b5bc39-88f6-4ddc-8e4b-2bb9c0c1567a",
    authority: "https://login.microsoftonline.com/4d5f34d3-d97b-40c7-8704-edff856d3654",
    redirectUri: window.location.origin
  },
  cache: {
    cacheLocation: "sessionStorage",
    storeAuthStateInCookie: false,
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        console.log("MSAL:", message);
      },
      logLevel: msal.LogLevel.Verbose,
      piiLoggingEnabled: false
    }
  }
};

const msalInstance = new msal.PublicClientApplication(msalConfig);

// Initialize MSAL
msalInstance.initialize().then(() => {
  console.log("MSAL initialized");
  
  // Check if user is already signed in
  const accounts = msalInstance.getAllAccounts();
  if (accounts.length > 0) {
    const account = accounts[0];
    document.getElementById("userWelcome").innerText = `Welcome, ${account.name}!`;
    document.getElementById("signInButton").style.display = "none";
    document.getElementById("signOutButton").style.display = "inline-block";
    document.getElementById("submitButton").disabled = false;
  }
}).catch((error) => {
  console.error("MSAL initialization failed:", error);
});

async function signIn() {
  try {
    
    console.log("Starting sign in with config:", msalConfig);
    
    const loginRequest = {
      scopes: ["openid", "profile", "email"],
      // Add domain hint to help with routing
      extraQueryParameters: {
        domain_hint: "RideshareDavid64.onmicrosoft.com"
      }
    };
    
    console.log("Login request:", loginRequest);
    
    const loginResponse = await msalInstance.loginPopup(loginRequest);
    
    // Log successful response details
    console.log("Login successful - Full response:", loginResponse);
    console.log("Account details:", loginResponse.account);
    console.log("ID token:", loginResponse.idToken);
    console.log("Scopes granted:", loginResponse.scopes);

    sessionStorage.setItem("id_token", loginResponse.idToken);
    const account = loginResponse.account;
    console.log(account);
    document.getElementById("userWelcome").innerText = `Welcome, ${account.name}!`;
    
    document.getElementById("signInButton").style.display = "none";
    document.getElementById("signOutButton").style.display = "inline-block";
    document.getElementById("submitButton").disabled = false;
  } catch (error) {
    console.error("Detailed error info:");
    console.error("Error name:", error.name);
    console.error("Error message:", error.message);
    console.error("Error code:", error.errorCode);
    console.error("Error description:", error.errorDesc);
    console.error("Correlation ID:", error.correlationId);
    console.error("Full error object:", error);
    alert("Sign in failed." + error.message);
  }
}

async function signOut() {
  try {
    await msalInstance.logoutPopup();
    sessionStorage.removeItem("id_token");
    document.getElementById("signInButton").style.display = "inline-block";
    document.getElementById("signOutButton").style.display = "none";   
    
    document.getElementById("question").value = "";
    document.getElementById("answer").value = "";
    document.getElementById("submitButton").disabled = true;
    document.getElementById("userWelcome").innerText = "Please sign in to ask a question.";
  } catch (error) {
    console.error("Sign out failed:", error);
  }
}

async function sendQuestion() {
  const idToken = sessionStorage.getItem("id_token");

  if(idToken === null) {
    alert("You must be signed in to ask a question.");  
    return;
  }

  const submitButton = document.getElementById("submitButton");
  submitButton.disabled = true;
  submitButton.innerText = "Please wait...";

  const selectedSource = document.querySelector('input[name="knowledgeSource"]:checked').value;

  const question = document.getElementById("question").value.trim();
  try {
    
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${idToken}`
      },
      body: JSON.stringify({question,
        knowledgeSource: selectedSource 
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    document.getElementById("answer").value = data.answer;
  } catch (error) {
    console.error("Error sending question:", error);
    alert("Something went wrong. Please try again.");
  } finally {
    submitButton.disabled = false;
    submitButton.innerText = "Submit";
  }
}