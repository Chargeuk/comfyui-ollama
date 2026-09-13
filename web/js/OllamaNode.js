import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.OllamaNode",
  aboutPageBadges: [
    {
      label: "ComfyUI-Ollama",
      url: "https://github.com/stavsap/comfyui-ollama",
      icon: "pi pi-github",
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["OllamaGenerate", "OllamaGenerateAdvance", "OllamaVision", "OllamaVts", "OllamaImageQuestionsVts", "OllamaSettingsVts", "OllamaConnectivityV2"].includes(nodeData.name)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const apiProviderWidget = this.widgets.find((w) => w.name === "api_provider");
        const apiKeyWidget = this.widgets.find((w) => w.name === "api_key");
        const modelWidget = this.widgets.find((w) => w.name === "model");
        let refreshButtonWidget = {};
        if (nodeData.name === "OllamaConnectivityV2") {
          refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");
        }

        const fetchModels = async (url, apiProvider, apiKey) => {
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
            throw new Error(response);
          }
        };

        const updateModels = async () => {
          refreshButtonWidget.name = "⏳ Fetching...";
          const url = urlWidget.value;
          const apiProvider = apiProviderWidget?.value ?? "ollama";
          const apiKey = apiKeyWidget?.value ?? "";
          const prevValue = modelWidget.value;
          modelWidget.value = "";
          modelWidget.options.values = [];

          let models = [];
          try {
            models = await fetchModels(url, apiProvider, apiKey);
          } catch (error) {
            console.error("Error fetching models:", error);
            app.extensionManager.toast.add({
              severity: "error",
              summary: "Ollama connection error",
              detail: "Make sure the selected model provider is running",
              life: 5000,
            });
            refreshButtonWidget.name = "🔄 Reconnect";
            this.setDirtyCanvas(true);
            return;
          }

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue; // stay on current.
          } else if (models.length > 0) {
            modelWidget.value = models[0]; // set first as default.
          }

          refreshButtonWidget.name = "🔄 Reconnect";
          this.setDirtyCanvas(true);
          console.debug("Updated modelWidget.value:", modelWidget.value);
        };

        urlWidget.callback = updateModels;
        if (apiProviderWidget) {
          apiProviderWidget.callback = updateModels;
        }
        if (apiKeyWidget) {
          apiKeyWidget.callback = updateModels;
        }
        refreshButtonWidget.callback = updateModels;

        const dummy = async () => {
          // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        };

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();
      };
    }
  },
});
