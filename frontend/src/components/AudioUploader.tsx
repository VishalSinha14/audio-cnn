import { useState, useRef, useCallback } from 'react';

interface AudioUploaderProps {
  onFileSelected: (file: File | Blob) => void;
  isLoading: boolean;
}

/**
 * Encode raw PCM Float32 samples into a WAV Blob that the backend can read.
 * This avoids sending webm/ogg from MediaRecorder, which soundfile can't decode.
 */
function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);             // chunk size
  view.setUint16(20, 1, true);              // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert float32 → int16
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export default function AudioUploader({ onFileSelected, isLoading }: AudioUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setSelectedFileName(file.name);
      setRecordingError(null);
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFileName(file.name);
      setRecordingError(null);
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const startRecording = useCallback(async () => {
    setRecordingError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 44100,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Convert recorded webm/ogg blob → proper WAV via AudioContext
        try {
          const recordedBlob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType });
          const arrayBuffer = await recordedBlob.arrayBuffer();

          const audioCtx = new AudioContext({ sampleRate: 44100 });
          audioContextRef.current = audioCtx;

          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

          // Mix down to mono
          const mono = audioBuffer.numberOfChannels === 1
            ? audioBuffer.getChannelData(0)
            : (() => {
                const ch0 = audioBuffer.getChannelData(0);
                const ch1 = audioBuffer.getChannelData(1);
                const mixed = new Float32Array(ch0.length);
                for (let i = 0; i < ch0.length; i++) {
                  mixed[i] = (ch0[i] + ch1[i]) / 2;
                }
                return mixed;
              })();

          const wavBlob = encodeWAV(mono, audioBuffer.sampleRate);
          setSelectedFileName(`Recording (${formatTime(recordingTime)})`);
          onFileSelected(wavBlob);

          audioCtx.close();
        } catch (err) {
          console.error('WAV conversion failed:', err);
          setRecordingError('Could not process recording. Try uploading a file instead.');
        }

        // Clean up microphone stream
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      };

      mediaRecorder.start(250); // collect chunks every 250ms
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone access error:', err);
      setRecordingError(
        'Could not access microphone. Please allow microphone permissions in your browser settings.'
      );
    }
  }, [onFileSelected, recordingTime]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>🎵</span> Audio Input
        </div>
        {selectedFileName && (
          <span className="card-badge">{selectedFileName}</span>
        )}
      </div>

      <div
        className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isRecording && fileInputRef.current?.click()}
      >
        <div className="upload-zone-content">
          {isRecording ? (
            <>
              <span className="upload-icon">🎙️</span>
              <div className="upload-text">
                <span className="recording-indicator" /> Recording... {formatTime(recordingTime)}
              </div>
              <div className="upload-hint">Click stop when you're done</div>
            </>
          ) : isLoading ? (
            <>
              <div className="loading-spinner" />
              <div className="loading-text">Classifying audio...</div>
            </>
          ) : (
            <>
              <span className="upload-icon">🎧</span>
              <div className="upload-text">Drag & drop an audio file here</div>
              <div className="upload-hint">or click to browse • WAV, MP3, FLAC, OGG</div>
            </>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          style={{ display: 'none' }}
          onChange={handleFileInput}
          id="audio-file-input"
        />
      </div>

      {recordingError && (
        <div style={{
          marginTop: '0.75rem',
          padding: '0.5rem 0.75rem',
          borderRadius: '0.5rem',
          background: 'rgba(255, 80, 80, 0.1)',
          border: '1px solid rgba(255, 80, 80, 0.25)',
          color: '#ff6b6b',
          fontSize: '0.85rem',
        }}>
          ⚠️ {recordingError}
        </div>
      )}

      <div className="upload-actions">
        {isRecording ? (
          <button className="btn btn-danger" onClick={stopRecording} id="stop-recording-btn">
            ⏹ Stop Recording
          </button>
        ) : (
          <>
            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              id="upload-btn"
            >
              📁 Upload File
            </button>
            <button
              className="btn btn-secondary"
              onClick={(e) => { e.stopPropagation(); startRecording(); }}
              disabled={isLoading}
              id="record-btn"
            >
              🎤 Record
            </button>
          </>
        )}
      </div>
    </div>
  );
}
