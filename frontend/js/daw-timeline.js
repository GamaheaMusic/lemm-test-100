/**
 * DAW-Style Timeline Module
 * Handles horizontal timeline with tracks, drag-and-drop, playback
 */

class DAWTimeline {
    constructor() {
        this.timeline = [];
        this.zoomLevel = 50; // pixels per second
        this.playheadPosition = 0;
        this.isPlaying = false;
        this.selectedClip = null;
        this.draggedClip = null;
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.audioBuffers = new Map(); // Store loaded audio buffers
        this.currentSources = []; // Track currently playing sources
        this.startTime = 0;
        this.pauseTime = 0;
        
        // DOM elements
        this.rulerCanvas = document.getElementById('rulerCanvas');
        this.musicTrackContent = document.getElementById('musicTrackContent');
        this.playhead = document.getElementById('playhead');
        this.currentTimeDisplay = document.getElementById('currentTime');
        this.totalDurationDisplay = document.getElementById('totalDuration');
        this.clipCountDisplay = document.getElementById('clipCount');
        
        // Buttons
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.rewindBtn = document.getElementById('rewindBtn');
        this.zoomInBtn = document.getElementById('zoomInBtn');
        this.zoomOutBtn = document.getElementById('zoomOutBtn');
        this.fitBtn = document.getElementById('fitBtn');
        
        this.setupEventListeners();
        this.initializeRuler();
    }
    
    setupEventListeners() {
        // Playback controls
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.rewindBtn.addEventListener('click', () => this.rewind());
        
        // Zoom controls
        this.zoomInBtn.addEventListener('click', () => this.zoomIn());
        this.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        this.fitBtn.addEventListener('click', () => this.fitToScreen());
        
        // Track click for playhead positioning
        this.musicTrackContent.addEventListener('click', (e) => this.handleTrackClick(e));
        
        // Window resize
        window.addEventListener('resize', () => this.redrawRuler());
    }
    
    initializeRuler() {
        const ctx = this.rulerCanvas.getContext('2d');
        this.rulerCanvas.width = this.rulerCanvas.offsetWidth;
        this.rulerCanvas.height = this.rulerCanvas.offsetHeight;
        this.redrawRuler();
    }
    
    redrawRuler() {
        const canvas = this.rulerCanvas;
        const ctx = canvas.getContext('2d');
        
        // Resize canvas to match container
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        const totalDuration = this.getTotalDuration();
        const width = Math.max(canvas.width, totalDuration * this.zoomLevel);
        
        // Clear canvas
        ctx.clearRect(0, 0, width, canvas.height);
        
        // Draw background
        ctx.fillStyle = '#2a2a2a';
        ctx.fillRect(0, 0, width, canvas.height);
        
        // Draw time markers
        ctx.fillStyle = '#aaa';
        ctx.font = '11px Arial';
        ctx.textAlign = 'center';
        
        const secondInterval = this.zoomLevel;
        const majorInterval = 5; // Major tick every 5 seconds
        
        for (let i = 0; i <= totalDuration; i++) {
            const x = i * secondInterval;
            
            if (i % majorInterval === 0) {
                // Major tick
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, canvas.height - 15);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
                
                // Time label
                ctx.fillText(this.formatTime(i), x, 15);
            } else {
                // Minor tick
                ctx.strokeStyle = '#444';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, canvas.height - 8);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
        }
    }
    
    loadTimeline(clips) {
        this.timeline = clips;
        this.renderClips();
        this.updateStats();
        this.redrawRuler();
    }
    
    renderClips() {
        // Clear tracks
        this.musicTrackContent.innerHTML = '';
        
        if (this.timeline.length === 0) {
            this.musicTrackContent.innerHTML = '<div class="empty-track-hint">Drop clips here or generate music above</div>';
            return;
        }
        
        // Render all clips on music track
        this.timeline.forEach((clip, index) => {
            const clipElement = this.createClipElement(clip, index);
            this.musicTrackContent.appendChild(clipElement);
        });
    }
    
    createClipElement(clip, index) {
        const div = document.createElement('div');
        div.className = 'track-clip';
        div.dataset.clipId = clip.clip_id;
        div.style.left = (clip.start_time * this.zoomLevel) + 'px';
        div.style.width = (clip.duration * this.zoomLevel) + 'px';
        
        // Gradient based on index for visual distinction
        const hue = (index * 40) % 360;
        div.style.background = `linear-gradient(135deg, hsl(${hue}, 70%, 60%) 0%, hsl(${hue + 30}, 70%, 50%) 100%)`;
        
        div.innerHTML = `
            <div class="clip-name">Clip ${index + 1}</div>
            <div class="clip-duration">${this.formatTime(clip.duration)}</div>
            <canvas class="clip-waveform" width="200" height="24"></canvas>
            <div class="clip-resize-handle left"></div>
            <div class="clip-resize-handle right"></div>
        `;
        
        // Make clip draggable
        div.draggable = true;
        div.addEventListener('dragstart', (e) => this.handleClipDragStart(e));
        div.addEventListener('dragend', (e) => this.handleClipDragEnd(e));
        div.addEventListener('click', (e) => this.selectClip(e, clip.clip_id));
        
        // Draw simple waveform
        this.drawWaveform(div.querySelector('.clip-waveform'), clip);
        
        return div;
    }
    
    drawWaveform(canvas, clip) {
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        
        // Simple pseudo-waveform for visual effect
        ctx.beginPath();
        const samples = 50;
        for (let i = 0; i < samples; i++) {
            const x = (canvas.width / samples) * i;
            const y = canvas.height / 2 + (Math.sin(i * 0.5) + Math.random() - 0.5) * (canvas.height / 3);
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }
    
    handleClipDragStart(e) {
        this.draggedClip = e.target;
        e.target.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    }
    
    handleClipDragEnd(e) {
        e.target.classList.remove('dragging');
        this.draggedClip = null;
    }
    
    handleTrackClick(e) {
        if (e.target.classList.contains('track-content')) {
            // Calculate time position
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const time = x / this.zoomLevel;
            this.setPlayheadPosition(time);
        }
    }
    
    selectClip(e, clipId) {
        e.stopPropagation();
        
        // Deselect all clips
        document.querySelectorAll('.track-clip').forEach(clip => {
            clip.classList.remove('selected');
        });
        
        // Select this clip
        e.currentTarget.classList.add('selected');
        this.selectedClip = clipId;
    }
    
    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }
    
    async play() {
        if (this.timeline.length === 0) {
            showStatus('No clips to play', 'warning');
            return;
        }
        
        this.isPlaying = true;
        this.playPauseBtn.innerHTML = '⏸️';
        
        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        // Start playing all clips from current position
        this.startTime = this.audioContext.currentTime - this.playheadPosition;
        
        for (const clip of this.timeline) {
            await this.playClip(clip);
        }
        
        this.animatePlayhead();
    }
    
    async playClip(clip) {
        try {
            // Build full URL for audio file
            const audioPath = clip.music_path || clip.mixed_path || clip.file_path;
            if (!audioPath) {
                console.error('No audio path found for clip:', clip);
                throw new Error('Audio file path is missing');
            }
            
            // Construct absolute URL
            let audioUrl;
            if (audioPath.startsWith('http')) {
                audioUrl = audioPath;
            } else if (audioPath.startsWith('/')) {
                audioUrl = `${API_BASE_URL}${audioPath}`;
            } else {
                audioUrl = `${API_BASE_URL}/${audioPath}`;
            }
            
            console.log('Loading audio from:', audioUrl);
            
            // Load audio buffer if not already loaded
            if (!this.audioBuffers.has(clip.clip_id)) {
                const response = await fetch(audioUrl);
                
                if (!response.ok) {
                    throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`);
                }
                
                const contentType = response.headers.get('content-type');
                console.log('Audio content-type:', contentType);
                
                const arrayBuffer = await response.arrayBuffer();
                console.log('Audio buffer size:', arrayBuffer.byteLength);
                
                const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
                this.audioBuffers.set(clip.clip_id, audioBuffer);
                console.log('Audio decoded successfully');
            }
            
            const buffer = this.audioBuffers.get(clip.clip_id);
            const source = this.audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(this.audioContext.destination);
            
            // Calculate when this clip should start
            const clipStartTime = this.startTime + clip.start_time;
            const now = this.audioContext.currentTime;
            
            // Only play if clip should be playing at current playhead position
            if (clipStartTime + clip.duration > now && clipStartTime <= now + this.playheadPosition) {
                const offset = Math.max(0, now - clipStartTime);
                const duration = Math.min(clip.duration - offset, clip.duration);
                
                if (duration > 0) {
                    source.start(Math.max(now, clipStartTime), offset, duration);
                    this.currentSources.push(source);
                }
            }
        } catch (error) {
            console.error(`Failed to play clip ${clip.clip_id}:`, error);
            showStatus(`Error playing audio: ${error.message}`, 'error');
        }
    }
    
    pause() {
        this.isPlaying = false;
        this.playPauseBtn.innerHTML = '▶️';
        
        // Stop all currently playing sources
        for (const source of this.currentSources) {
            try {
                source.stop();
            } catch (e) {
                // Source may have already ended
            }
        }
        this.currentSources = [];
        
        this.pauseTime = this.playheadPosition;
    }
    
    stop() {
        this.isPlaying = false;
        this.playPauseBtn.innerHTML = '▶️';
        
        // Stop all currently playing sources
        for (const source of this.currentSources) {
            try {
                source.stop();
            } catch (e) {
                // Source may have already ended
            }
        }
        this.currentSources = [];
        
        this.setPlayheadPosition(0);
        this.pauseTime = 0;
    }
    
    rewind() {
        this.setPlayheadPosition(0);
    }
    
    animatePlayhead() {
        if (!this.isPlaying) return;
        
        const totalDuration = this.getTotalDuration();
        this.playheadPosition += 0.1; // Advance by 0.1 seconds
        
        if (this.playheadPosition >= totalDuration) {
            this.stop();
            return;
        }
        
        this.setPlayheadPosition(this.playheadPosition);
        requestAnimationFrame(() => this.animatePlayhead());
    }
    
    setPlayheadPosition(time) {
        this.playheadPosition = Math.max(0, time);
        this.playhead.style.left = (this.playheadPosition * this.zoomLevel) + 'px';
        this.currentTimeDisplay.textContent = this.formatTime(this.playheadPosition);
    }
    
    zoomIn() {
        this.zoomLevel = Math.min(200, this.zoomLevel * 1.5);
        this.renderClips();
        this.redrawRuler();
        this.setPlayheadPosition(this.playheadPosition);
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(20, this.zoomLevel / 1.5);
        this.renderClips();
        this.redrawRuler();
        this.setPlayheadPosition(this.playheadPosition);
    }
    
    fitToScreen() {
        const totalDuration = this.getTotalDuration();
        if (totalDuration > 0) {
            const containerWidth = this.musicTrackContent.offsetWidth;
            this.zoomLevel = Math.max(20, containerWidth / totalDuration * 0.9);
            this.renderClips();
            this.redrawRuler();
            this.setPlayheadPosition(this.playheadPosition);
        }
    }
    
    updateStats() {
        const totalDuration = this.getTotalDuration();
        this.totalDurationDisplay.textContent = this.formatTime(totalDuration);
        this.clipCountDisplay.textContent = `${this.timeline.length} clip${this.timeline.length !== 1 ? 's' : ''}`;
    }
    
    getTotalDuration() {
        if (this.timeline.length === 0) return 60; // Default 60s for empty timeline
        
        let maxEnd = 0;
        this.timeline.forEach(clip => {
            const end = clip.start_time + clip.duration;
            if (end > maxEnd) maxEnd = end;
        });
        
        return Math.max(60, maxEnd + 10); // Add 10s padding
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 10);
        return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
    }
}

// Export for use in app.js
window.DAWTimeline = DAWTimeline;
