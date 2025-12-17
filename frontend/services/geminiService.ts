
import { GoogleGenAI, Type } from "@google/genai";
import { TelemetryData, RiskAnalysis, RCADataset } from '../types';

// Always initialize GoogleGenAI with a named parameter for apiKey
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const generateRCAFeedback = async (
  telemetry: TelemetryData, 
  analysis: RiskAnalysis
): Promise<RCADataset> => {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      // Fixed: Property 'min' does not exist on type 'JSON'. Use JSON.stringify instead.
      contents: `Perform Manufacturing Root Cause Analysis (RCA) and Corrective/Preventive Action (CAPA) for the following vehicle failure risk state:
      - Risk Band: ${analysis.band}
      - Final Risk Score: ${analysis.finalRisk.toFixed(2)}
      - Telemetry: ${JSON.stringify(telemetry)}
      - Decision Trace: ${analysis.decisionTrace.join(', ')}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            rootCause: { type: Type.STRING },
            remedy: { type: Type.STRING },
            capa: { type: Type.STRING }
          },
          // Use propertyOrdering in responseSchema as per @google/genai guidelines
          propertyOrdering: ["rootCause", "remedy", "capa"]
        }
      }
    });

    // Access the text property directly (not as a function)
    const text = response.text;
    if (!text) {
      throw new Error("Received empty response from Gemini API");
    }

    return JSON.parse(text);
  } catch (error) {
    console.error("Gemini RCA Error:", error);
    return {
      rootCause: "Unspecified mechanical failure due to parameter drift.",
      remedy: "Immediate inspection and thermal calibration.",
      capa: "Implement stricter tolerance filters in the telemetry gateway."
    };
  }
};
