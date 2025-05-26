import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import { Suspense, useState, useEffect } from "react";
import { Avatar } from "./Avatar";
import TalkToAIScreen from "./TalkToAIScreen";
import { useResponsiveStyles } from "../styles/responsive";
import { View } from "react-native";

interface ThreeAvatarProps {
  setActiveTab: (tab: string) => void;
}

export default function ThreeAvatar({ setActiveTab }: ThreeAvatarProps) {
  const [isTalking, setIsTalking] = useState(false);
  const [audioData, setAudioData] = useState<string | null>(null);
  const styles = useResponsiveStyles();

  return (
    <View style={styles.containerStyle}>
      <View style={styles.canvasStyle}>
        <Canvas camera={{ position: [-0.5, 2.5, 5], fov: 35 }}>
          <ambientLight intensity={1} color={"#ffffff"} />
          <pointLight position={[0, 3, 0]} intensity={5} color={"#ffddb1"} />

          <Suspense fallback={null}>
            <Avatar
              position={[-0.5, -1.0, 0]}
              scale={2}
              isTalking={isTalking}
              audio={audioData}
              setIsTalking={setIsTalking}
            />
            <ContactShadows
              position={[0, -1.2, 0]}
              opacity={0.6}
              scale={10}
              blur={2.5}
              far={4}
              frames={1}
              resolution={512}
              color="#000000"
            />
          </Suspense>
          <OrbitControls
            target={[0.3, 1.2, 0]}
            enableZoom={false}
            enablePan={false}
          />
        </Canvas>
      </View>

      <View style={styles.chatContainerStyle}>
        <TalkToAIScreen
          setActiveTab={setActiveTab}
          setIsTalking={setIsTalking}
          setAudioData={setAudioData}
          isDesktopView={styles.isDesktopView}
        />
      </View>

      {styles.isDesktopView && (
        <View style={styles.voiceButtonStyle}>
          <TalkToAIScreen
            setActiveTab={setActiveTab}
            setIsTalking={setIsTalking}
            setAudioData={setAudioData}
            isDesktopView={styles.isDesktopView}
            voiceOnlyMode={true}
          />
        </View>
      )}
    </View>
  );
}
