document.addEventListener("DOMContentLoaded", function () {
  let currentLang = "en"; // Default language

  // Fetch translations from the JSON file
  fetch("../static/translations.json")
    .then(response => response.json())
    .then(data => {
      window.translations = data;
      applyTranslations(currentLang);
    })
    .catch(err => console.error("Error loading translations:", err));

  // Function to update elements with translations
  function applyTranslations(lang) {
    document.querySelectorAll("[data-translate-key]").forEach(el => {
      const key = el.getAttribute("data-translate-key");
      if (window.translations && window.translations[lang] && window.translations[lang][key]) {
        // For form inputs like the submit button, update the "value"
        if (el.tagName === "INPUT" && el.type === "submit") {
          el.value = window.translations[lang][key];
        } else {
          el.textContent = window.translations[lang][key];
        }
      }
    });
  }

  // Listen for changes on the language selector
  const langSelector = document.getElementById("langSelector");
  if (langSelector) {
    langSelector.addEventListener("change", function () {
      currentLang = this.value;
      applyTranslations(currentLang);
    });
  }
});
