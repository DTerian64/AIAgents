async function sendQuestion() {
  const button = document.getElementById("submitButton");
  button.disabled = true;
  button.innerText = "Please wait...";

  const question = document.getElementById("question").value;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
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
