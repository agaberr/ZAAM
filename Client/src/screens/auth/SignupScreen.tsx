import React, { useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { TextInput, Button, Title, HelperText } from 'react-native-paper';

const SignupScreen = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  const handleSignup = () => {
    // Basic validation
    if (!email || !password || !confirmPassword) {
      setError('Please fill in all fields');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    // Navigate to user data screen with email and password
    navigation.navigate('UserData', { email, password });
  };

  return (
    <ScrollView style={styles.container}>
      <Title style={styles.title}>Create Account</Title>

      <TextInput
        label="Email"
        value={email}
        onChangeText={setEmail}
        mode="outlined"
        style={styles.input}
        keyboardType="email-address"
        autoCapitalize="none"
      />

      <TextInput
        label="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        mode="outlined"
        style={styles.input}
      />

      <TextInput
        label="Confirm Password"
        value={confirmPassword}
        onChangeText={setConfirmPassword}
        secureTextEntry
        mode="outlined"
        style={styles.input}
      />

      {error ? <HelperText type="error">{error}</HelperText> : null}

      <Button mode="contained" onPress={handleSignup} style={styles.button}>
        Next
      </Button>

      <Button mode="text" onPress={() => navigation.navigate('Login')} style={styles.loginButton}>
        Already have an account? Log In
      </Button>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    marginTop: 60,
    marginBottom: 30,
    textAlign: 'center',
  },
  input: {
    marginBottom: 15,
    backgroundColor: 'white',
  },
  button: {
    marginTop: 10,
    marginBottom: 15,
    paddingVertical: 6,
  },
  loginButton: {
    marginBottom: 20,
  },
});

export default SignupScreen;
