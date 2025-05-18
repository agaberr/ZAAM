import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import { Suspense, useState } from "react";
import { Avatar } from "./Avatar";
import UI from "./UI";

export default function ThreeAvatar({ setActiveTab }) {
  const [isTalking, setIsTalking] = useState(false);
  const [audioData, setAudioData] = useState<string | null>(null);
  return (
    <div
      style={{
        height: "100vh",
        background: "#2c1f16",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <Canvas camera={{ position: [-1.5, 2.5, 6], fov: 35 }}>
        <ambientLight intensity={0.7} color={"#ffddb1"} />
        <pointLight position={[-1.5, 3.5, 0]} intensity={5} color={"#ffddb1"} />

        <Suspense fallback={null}>
          <Avatar
            position={[-1.5, -1.0, 0]}
            scale={2}
            isTalking={isTalking}
            audio="./OUT.wav"
            setIsTalking={setIsTalking}
          />
          <ContactShadows
            position={[0, -1.2, 0]}
            opacity={0.6}
            scale={10}
            blur={2.5}
            far={4}
          />
        </Suspense>

        <OrbitControls
          target={[0.3, 1.2, 0]}
          enableZoom={false}
          enablePan={false}
        />
      </Canvas>

      <div
        style={{
          position: "absolute",
          bottom: 0,
          right: 120,
          width: "40%",
          height: "100%",
          zIndex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <UI
          setActiveTab={setActiveTab}
          setIsTalking={setIsTalking}
          setAudioData={setAudioData}
        />
      </div>
    </div>
  );
}
