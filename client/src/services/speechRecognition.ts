// import * as Speech from 'expo-speech';
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
    // Speech functionality commented out due to compatibility issues
    console.log('Speech disabled:', text);
    return Promise.resolve();
    // return Speech.speak(text, {
    //   language: 'en-US',
    //   pitch: 1.0,
    //   rate: 0.9,
    //   ...options,
    // });
  };

  const stopSpeaking = () => {
    // Speech functionality commented out due to compatibility issues
    console.log('Stop speaking disabled');
    return Promise.resolve();
    // return Speech.stop();
  };

  return {
    startListening,
    stopListening,
    speak,
    stopSpeaking,
  };
};
