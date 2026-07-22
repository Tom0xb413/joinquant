(() => {
  const button = document.getElementById("run-once");
  if (!button) return;
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "同步中…";
    try {
      const response = await fetch("/api/run-once", { method: "POST" });
      if (!response.ok) throw new Error("unauthorized or failed");
      window.location.reload();
    } catch (error) {
      button.textContent = "同步失败";
      button.disabled = false;
    }
  });
})();
