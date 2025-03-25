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
          reservedClientId: "your-ios-client-id"
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
      ],
      // Commenting out expo-speech due to compatibility issues
      /*[
        "expo-speech",
        {
          microphonePermission: "The app needs access to your microphone for voice interactions."
        }
      ]*/
    ],
    extra: {
      apiUrl: process.env.API_URL || "https://zaam-mj7u.onrender.com/api",
      eas: {
        projectId: "your-project-id"
      }
    }
  }
};
