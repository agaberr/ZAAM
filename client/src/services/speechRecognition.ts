import * as Speech from 'expo-speech';
import Voice, { SpeechResultsEvent } from '@react-native-voice/voice';

export const useSpeechRecognition = () => {
  const startListening = () => {
    return new Promise((resolve, reject) => {
      Voice.start('en-US')
        .then(() => resolve(true))
        .catch(reject);
    });
  };

  const stopListening = () => {
    return Voice.stop();
  };

  const speak = (text: string, options = {}) => {
    return Speech.speak(text, {
      language: 'en-US',
      pitch: 1.0,
      rate: 0.9,
      ...options,
    });
  };

  const stopSpeaking = () => {
    return Speech.stop();
  };

  return {
    startListening,
    stopListening,
    speak,
    stopSpeaking,
  };
};
