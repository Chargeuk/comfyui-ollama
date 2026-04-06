import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.OllamaNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["OllamaGenerate", "OllamaGenerateAdvance", "OllamaVision", "OllamaVts", "OllamaImageQuestionsVts", "OllamaSettingsVts"].includes(nodeData.name) ) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const apiProviderWidget = this.widgets.find((w) => w.name === "api_provider");
        const apiKeyWidget = this.widgets.find((w) => w.name === "api_key");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        const fetchModels = async (url, apiProvider, apiKey) => {
          try {
            const response = await fetch("/ollama/get_models", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url,
                api_provider: apiProvider,
                api_key: apiKey,
              }),
            });

            if (response.ok) {
              const models = await response.json();
              console.debug("Fetched models:", models);
              return models;
            } else {
              console.error(`Failed to fetch models: ${response.status}`);
              return [];
            }
          } catch (error) {
            console.error(`Error fetching models`, error);
            return [];
          }
        };

        const updateModels = async () => {
          const url = urlWidget.value;
          const apiProvider = apiProviderWidget?.value ?? "ollama";
          const apiKey = apiKeyWidget?.value ?? "";
          const prevValue = modelWidget.value
          modelWidget.value = ''
          modelWidget.options.values = []

          const models = await fetchModels(url, apiProvider, apiKey);

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue; // stay on current.
          } else if (models.length > 0) {
            modelWidget.value = models[0]; // set first as default.
          }

          console.debug("Updated modelWidget.value:", modelWidget.value);
        };

        urlWidget.callback = updateModels;
        if (apiProviderWidget) {
          apiProviderWidget.callback = updateModels;
        }
        if (apiKeyWidget) {
          apiKeyWidget.callback = updateModels;
        }

        const dummy = async () => {
          // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        }

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();
      };
    }
  },
});
