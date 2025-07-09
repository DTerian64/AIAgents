const msalConfig = {
  auth: {
    clientId: "177da031-26fa-448a-8521-1d9bedde86d3",
    authority: "https://login.microsoftonline.com/4d5f34d3-d97b-40c7-8704-edff856d3654",
    redirectUri: window.location.origin
  }
};

const msalInstance = new msal.PublicClientApplication(msalConfig);

async function signIn() {
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
}

async function signOut() {
  await msalInstance.logoutPopup(); // This will clear the session storage
  sessionStorage.removeItem("id_token");
  document.getElementById("signInButton").style.display = "inline-block";
  document.getElementById("signOutButton").style.display = "none";   
  
  document.getElementById("question").value = "";
  document.getElementById("answer").value = "";
  document.getElementById("submitButton").disabled = true;
  document.getElementById("userWelcome").innerText = "Please sign in to ask a question.";
}      

async function sendQuestion() {

  const idToken = sessionStorage.getItem("id_token");

  if(idToken === null) {
    alert("You must be signed in to ask a question.");  
    return;
  }

  const button = document.getElementById("submitButton");
  button.disabled = true;
  button.innerText = "Please wait...";

  const question = document.getElementById("question").value;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json",
                "Authorization": `Bearer ${idToken}`},
      body: JSON.stringify({question})
    });
    const data = await response.json();
    document.getElementById("answer").value = data.answer; // If using <textarea>
  } catch (error) {
    console.error(error);
    alert("Something went wrong. Please try again.");
  } finally {
    button.disabled = false;
    button.innerText = "Submit";
  }
}
