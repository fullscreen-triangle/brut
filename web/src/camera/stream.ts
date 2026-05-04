import { log } from '../util/log';

export interface CameraStream {
  video: HTMLVideoElement;
  stream: MediaStream;
  width: number;
  height: number;
}

export async function startCamera(video: HTMLVideoElement): Promise<CameraStream> {
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
      facingMode: 'user',
    },
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error('video element error'));
  });
  await video.play();

  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings();
  const width = settings.width ?? video.videoWidth;
  const height = settings.height ?? video.videoHeight;

  log(`camera ready: ${width}x${height}@${settings.frameRate ?? '?'}fps device="${track.label}"`);

  return { video, stream, width, height };
}

export function stopCamera(cs: CameraStream): void {
  cs.stream.getTracks().forEach((t) => t.stop());
  cs.video.srcObject = null;
}
