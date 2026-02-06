// src/main.tsx – Main app entry with video + gesture overlay integration
import React, { useRef } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import GestureOverlay from '@/integrations/gesture-recognition/GestureOverlay.tsx';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root')!);

const MainApp = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    startVideo();
  }, []);

  return (
    <>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="fixed inset-0 w-full h-full object-cover z-0"
      />
      <div className="relative z-10">
        <App />
      </div>
      <GestureOverlay videoRef={videoRef} />
    </>
  );
};

root.render(
  <React.StrictMode>
    <MainApp />
  </React.StrictMode>
);
