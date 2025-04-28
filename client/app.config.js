export default {
  expo: {
    name: "Alzheimer's AI Companion",
    slug: "alzheimer-ai-companion",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "light",
    splash: {
      image: "./assets/splash.png",
      resizeMode: "contain",
      backgroundColor: "#6200ee"
    },
    assetBundlePatterns: [
      "**/*"
    ],
    ios: {
      supportsTablet: true,
      bundleIdentifier: "com.yourcompany.alzheimerai",
      config: {
        googleSignIn: {
          reservedClientId: "com.googleusercontent.apps.485643114726-0joadin731ltorui0db3v6dma86rjbvb"
        }
      }
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#6200ee"
      },
      package: "com.yourcompany.alzheimerai",
      permissions: [
        "CAMERA",
        "RECORD_AUDIO",
        "MODIFY_AUDIO_SETTINGS",
        "READ_CALENDAR",
        "WRITE_CALENDAR"
      ]
    },
    web: {
      favicon: "./assets/favicon.png"
    },
    plugins: [
      [
        "expo-calendar",
        {
          calendarPermission: "The app needs to access your calendar to add and manage reminders."
        }
      ]
    ],
    extra: {
      apiUrl: "http://localhost:5000/api",
      googleWebClientId: "485643114726-0joadin731ltorui0db3v6dma86rjbvb.apps.googleusercontent.com",
      eas: {
        projectId: "your-project-id"
      }
    },
    scheme: "alzheimer-ai-companion",
    owner: "zaam"
  }
};
