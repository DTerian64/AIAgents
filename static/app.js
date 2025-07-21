const msalConfig = {
  auth: {
    clientId: "177da031-26fa-448a-8521-1d9bedde86d3",
    authority: "https://login.microsoftonline.com/4d5f34d3-d97b-40c7-8704-edff856d3654",
    redirectUri: "https://david64aichat-bsecexfhgggmaghv.westus2-01.azurewebsites.net"
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
    const loginResponse = await msalInstance.loginPopup({
      scopes: ["openid", "profile", "email"]
    });
    console.log("ID token:", loginResponse.idToken);
    sessionStorage.setItem("id_token", loginResponse.idToken);

    const account = loginResponse.account;
    console.log(account);
    document.getElementById("userWelcome").innerText = `Welcome, ${account.name}!`;
    
    document.getElementById("signInButton").style.display = "none";
    document.getElementById("signOutButton").style.display = "inline-block";
    document.getElementById("submitButton").disabled = false;
  } catch (error) {
    console.error("Sign in failed:", error);
    alert("Sign in failed. Please try again.");
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