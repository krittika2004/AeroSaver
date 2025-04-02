document.addEventListener("DOMContentLoaded", function () {
    const toggleButton = document.getElementById("theme-toggle");
    const body = document.body;

    // Apply saved theme on page load
    if (localStorage.getItem("theme") === "dark") {
        body.classList.add("dark-mode");
        toggleButton.textContent = "☀️ Light Mode";
    }

    toggleButton.addEventListener("click", () => {
        body.classList.toggle("dark-mode");

        if (body.classList.contains("dark-mode")) {
            localStorage.setItem("theme", "dark");
            toggleButton.textContent = "☀️ Light Mode";
        } else {
            localStorage.setItem("theme", "light");
            toggleButton.textContent = "🌙 Dark Mode";
        }
    });
});
