'use client'

import { forwardRef, Suspense, useImperativeHandle, useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, View as ViewImpl, useGLTF, useAnimations } from '@react-three/drei'
import dynamic from 'next/dynamic'
import tunnel from 'tunnel-rat'

// Create tunnel for multi-view support
const r3f = tunnel()

// Three wrapper component
const Three = ({ children }) => {
  return <r3f.In>{children}</r3f.In>
}

// Scene component
const SceneComponent = ({ ...props }) => {
  return (
    <Canvas {...props}>
      <r3f.Out />
    </Canvas>
  )
}

// Dynamic import to prevent SSR issues
const Scene = dynamic(() => Promise.resolve(SceneComponent), { ssr: false })

// Lighting and camera setup
const SceneSetup = ({ 
  backgroundColor, 
  cameraPosition = [0, 0, 6], 
  cameraFov = 40,
  ambientIntensity = 0.5,
  pointLightIntensity = 1
}) => (
  <Suspense fallback={null}>
    {backgroundColor && <color attach='background' args={[backgroundColor]} />}
    <ambientLight intensity={ambientIntensity} />
    <pointLight position={[20, 30, 10]} intensity={pointLightIntensity} />
    <pointLight position={[-10, -10, -10]} color='blue' intensity={pointLightIntensity * 0.5} />
    <PerspectiveCamera makeDefault fov={cameraFov} position={cameraPosition} />
  </Suspense>
)

// Animated GLB Model Component
const AnimatedModel = ({ 
  modelPath, 
  autoRotate = false,
  rotationSpeed = 1,
  scale = 1, 
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  playAnimation = true,
  animationIndex = 0,
  ...props 
}) => {
  const group = useRef()
  const { scene, animations } = useGLTF(modelPath)
  const { actions, names } = useAnimations(animations, group)

  // Play animations automatically
  useEffect(() => {
    if (playAnimation && actions && names.length > 0) {
      const animationName = names[animationIndex] || names[0]
      const action = actions[animationName]
      if (action) {
        action.reset().play()
      }
      
      // Play all animations if you want
      // Object.values(actions).forEach(action => action.reset().play())
    }
  }, [actions, names, playAnimation, animationIndex])

  // Auto-rotation
  useFrame((state, delta) => {
    if (autoRotate && group.current) {
      group.current.rotation.y += delta * rotationSpeed
    }
  })

  return (
    <group ref={group}>
      <primitive 
        object={scene} 
        scale={scale}
        position={position}
        rotation={rotation}
        {...props} 
      />
    </group>
  )
}

// View component for multi-view setups
const View = forwardRef(({ children, orbit, ...props }, ref) => {
  const localRef = useRef(null)
  useImperativeHandle(ref, () => localRef.current)

  return (
    <>
      <div ref={localRef} {...props} />
      <Three>
        <ViewImpl track={localRef}>
          {children}
          {orbit && <OrbitControls />}
        </ViewImpl>
      </Three>
    </>
  )
})
View.displayName = 'View'

// Layout wrapper for fixed canvas background
const Layout = ({ children }) => {
  const ref = useRef()

  return (
    <div
      ref={ref}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'auto',
        touchAction: 'auto',
      }}
    >
      {children}
      <Scene
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          pointerEvents: 'none',
        }}
        eventSource={ref}
        eventPrefix='client'
      />
    </div>
  )
}

// 🎯 MAIN COMPONENT - Super Simple GLB Viewer
export const GLBViewer = ({ 
  modelPath,
  
  // Visual settings
  backgroundColor,
  
  // Camera settings
  cameraPosition = [0, 0, 6],
  cameraFov = 40,
  
  // Controls
  enableOrbit = true,
  autoRotate = false,
  rotationSpeed = 1,
  
  // Model settings
  modelScale = 1,
  modelPosition = [0, 0, 0],
  modelRotation = [0, 0, 0],
  
  // Animation settings
  playAnimation = true,
  animationIndex = 0,
  
  // Lighting
  ambientIntensity = 0.5,
  pointLightIntensity = 1,
  
  // Container settings
  className = '',
  style = {},
  
  ...props 
}) => {
  return (
    <div 
      className={className} 
      style={{ 
        width: '100%', 
        height: '100%', 
        position: 'relative',
        ...style 
      }}
    >
      <Canvas camera={{ position: cameraPosition, fov: cameraFov }}>
        <SceneSetup 
          backgroundColor={backgroundColor}
          cameraPosition={cameraPosition}
          cameraFov={cameraFov}
          ambientIntensity={ambientIntensity}
          pointLightIntensity={pointLightIntensity}
        />
        <Suspense fallback={null}>
          <AnimatedModel 
            modelPath={modelPath}
            autoRotate={autoRotate}
            rotationSpeed={rotationSpeed}
            scale={modelScale}
            position={modelPosition}
            rotation={modelRotation}
            playAnimation={playAnimation}
            animationIndex={animationIndex}
            {...props}
          />
        </Suspense>
        {enableOrbit && <OrbitControls />}
      </Canvas>
    </div>
  )
}

// Preload utility
export const preloadGLB = (modelPath) => {
  useGLTF.preload(modelPath)
}

// Export all components for advanced usage
export { View, Scene, Layout, Three, AnimatedModel, SceneSetup }

// Default export
export default GLBViewer
