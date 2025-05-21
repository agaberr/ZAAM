import { useAnimations, useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import React, { useEffect, useRef, useState, useMemo } from "react";
import * as THREE from "three";

export function Avatar(props) {
  const { nodes, materials, scene } = useGLTF("/models/MAN_LAST.glb");
  const { animations } = useGLTF("/models/animation.glb");

  const group = useRef();
  const { actions, mixer } = useAnimations(animations, group);
  const audioRef = useRef(null);
  const debugRef = useRef({ lastLogTime: 0 });

  const [lipsync, setLipsync] = useState();
  const [blink, setBlink] = useState(false);
  const [winkLeft, setWinkLeft] = useState(false);
  const [winkRight, setWinkRight] = useState(false);
  const [audioStatus, setAudioStatus] = useState("idle");

  const lipsyncSettings = useRef({
    amplifier: 5.0,
    mouthOpenSpeed: 0.5,
  });

  const setupMode = false;

  const defaultAnimation =
    animations.find((a) => a.name === "Circle")?.name ||
    animations[0]?.name ||
    "";
  const idleAnimation =
    animations.find((a) => a.name === "Idle")?.name ||
    animations[0]?.name ||
    "";
  const talkAnimation =
    animations.find((a) => a.name === "Talk1")?.name ||
    animations[0]?.name ||
    "";

  const [animation, setAnimation] = useState(defaultAnimation);

  useEffect(() => {
    if (!animation || !actions[animation]) return;
    actions[animation]
      .reset()
      .fadeIn(mixer.stats.actions.inUse === 0 ? 0 : 0.5)
      .play();
    return () => {
      actions[animation]?.fadeOut(0.5);
    };
  }, [animation, actions, mixer.stats.actions.inUse]);

  useEffect(() => {
    if (!props.isTalking || !props.audio) return;
    setAnimation(idleAnimation);

    const listener = new THREE.AudioListener();
    const sound = new THREE.Audio(listener);
    const audioLoader = new THREE.AudioLoader();

    audioRef.current = sound;

    setAudioStatus("loading");
    console.log("Loading audio...");

    audioLoader.load(
      props.audio,
      (buffer) => {
        console.log("Audio loaded successfully, preparing to play");
        setAudioStatus("loaded");

        sound.setBuffer(buffer);
        sound.setLoop(false);
        sound.setVolume(1.0);

        const analyser = new THREE.AudioAnalyser(sound, 64);

        sound.onEnded = () => {
          console.log("Audio playback ended");
          setAudioStatus("ended");
          setLipsync(undefined);
          setAnimation(idleAnimation);
        };

        try {
          console.log("Starting audio playback");
          sound.play();
          setAudioStatus("playing");

          setLipsync(() => ({
            getVolume: () => {
              if (!analyser) return 0;
              const frequencies = analyser.getFrequencyData();
              const speechRange = frequencies.slice(5, 30);
              const average =
                speechRange.reduce((a, b) => a + b, 0) / speechRange.length;
              return Math.pow(average / 256, 0.8);
            },
          }));

          setAnimation(talkAnimation);
        } catch (error) {
          console.error("Error playing audio:", error);
          setAudioStatus("error");
        }
      },
      undefined,
      (error) => {
        console.error("Error loading audio:", error);
        setAudioStatus("error");
      }
    );

    props.setIsTalking(false);

    return () => {
      console.log("Cleaning up audio");
      if (audioRef.current) {
        audioRef.current.stop();
        audioRef.current = null;
      }
      setLipsync(undefined);
      setAnimation(defaultAnimation);
      setAudioStatus("idle");
    };
  }, [props.isTalking, props.audio]);

  const throttledLog = (message, values) => {
    const now = Date.now();
    if (now - debugRef.current.lastLogTime > 1000) {
      console.log(message, values);
      debugRef.current.lastLogTime = now;
    }
  };

  const lerpMorphTarget = (target, value, speed = 2) => {
    scene.traverse((child) => {
      if (child.isSkinnedMesh && child.morphTargetDictionary) {
        const index = child.morphTargetDictionary[target];
        if (
          index === undefined ||
          child.morphTargetInfluences[index] === undefined
        )
          return;

        child.morphTargetInfluences[index] = THREE.MathUtils.lerp(
          child.morphTargetInfluences[index],
          value,
          speed
        );
      }
    });
  };

  useFrame(() => {
    if (!setupMode) {
      if (lipsync && (props.isTalking || audioStatus === "playing")) {
        const rawVolume = lipsync.getVolume();

        const amplifiedVolume = Math.min(
          rawVolume * lipsyncSettings.current.amplifier,
          1
        );

        const jitter = Math.random() * 0.1;
        const finalVolume = Math.min(amplifiedVolume + jitter, 1);

        throttledLog("Lipsync values:", {
          raw: rawVolume.toFixed(2),
          amplified: finalVolume.toFixed(2),
          status: audioStatus,
        });

        lerpMorphTarget(
          "mouthOpen",
          finalVolume,
          lipsyncSettings.current.mouthOpenSpeed
        );
        lerpMorphTarget(
          "mouthSmile",
          finalVolume * 0.6,
          lipsyncSettings.current.mouthOpenSpeed * 0.8
        );
      } else {
        lerpMorphTarget("mouthOpen", 0, 0.1);
        lerpMorphTarget("mouthSmile", 0, 0.1);
      }
    }

    lerpMorphTarget("eyeBlinkLeft", blink || winkLeft ? 1 : 0, 0.5);
    lerpMorphTarget("eyeBlinkRight", blink || winkRight ? 1 : 0, 0.5);
  });

  return (
    <group {...props} dispose={null} ref={group}>
      <primitive object={nodes.Hips} />
      <skinnedMesh
        name="EyeLeft"
        geometry={nodes.EyeLeft.geometry}
        material={materials.Wolf3D_Eye}
        skeleton={nodes.EyeLeft.skeleton}
        morphTargetDictionary={nodes.EyeLeft.morphTargetDictionary}
        morphTargetInfluences={nodes.EyeLeft.morphTargetInfluences}
      />
      <skinnedMesh
        name="EyeRight"
        geometry={nodes.EyeRight.geometry}
        material={materials.Wolf3D_Eye}
        skeleton={nodes.EyeRight.skeleton}
        morphTargetDictionary={nodes.EyeRight.morphTargetDictionary}
        morphTargetInfluences={nodes.EyeRight.morphTargetInfluences}
      />
      <skinnedMesh
        name="Wolf3D_Head"
        geometry={nodes.Wolf3D_Head.geometry}
        material={materials.Wolf3D_Skin}
        skeleton={nodes.Wolf3D_Head.skeleton}
        morphTargetDictionary={nodes.Wolf3D_Head.morphTargetDictionary}
        morphTargetInfluences={nodes.Wolf3D_Head.morphTargetInfluences}
      />
      <skinnedMesh
        name="Wolf3D_Teeth"
        geometry={nodes.Wolf3D_Teeth.geometry}
        material={materials.Wolf3D_Teeth}
        skeleton={nodes.Wolf3D_Teeth.skeleton}
        morphTargetDictionary={nodes.Wolf3D_Teeth.morphTargetDictionary}
        morphTargetInfluences={nodes.Wolf3D_Teeth.morphTargetInfluences}
      />
      <skinnedMesh
        geometry={nodes.Wolf3D_Glasses.geometry}
        material={materials.Wolf3D_Glasses}
        skeleton={nodes.Wolf3D_Glasses.skeleton}
      />
      <skinnedMesh
        geometry={nodes.Wolf3D_Headwear.geometry}
        material={materials.Wolf3D_Headwear}
        skeleton={nodes.Wolf3D_Headwear.skeleton}
      />
      <skinnedMesh
        geometry={nodes.Wolf3D_Body.geometry}
        material={materials.Wolf3D_Body}
        skeleton={nodes.Wolf3D_Body.skeleton}
      />
      <skinnedMesh
        geometry={nodes.Wolf3D_Outfit_Bottom.geometry}
        material={materials.Wolf3D_Outfit_Bottom}
        skeleton={nodes.Wolf3D_Outfit_Bottom.skeleton}
      />
      <skinnedMesh
        geometry={nodes.Wolf3D_Outfit_Footwear.geometry}
        material={materials.Wolf3D_Outfit_Footwear}
        skeleton={nodes.Wolf3D_Outfit_Footwear.skeleton}
      />
      <skinnedMesh
        geometry={nodes.Wolf3D_Outfit_Top.geometry}
        material={materials.Wolf3D_Outfit_Top}
        skeleton={nodes.Wolf3D_Outfit_Top.skeleton}
      />
    </group>
  );
}

useGLTF.preload("/models/MAN_LAST.glb");
useGLTF.preload("/models/animation.glb");
