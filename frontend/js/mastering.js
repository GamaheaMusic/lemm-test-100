/**
 * Advanced Mastering Module
 * Handles EQ, compression, limiting, and preset application
 */

class MasteringController {
    constructor() {
        this.modal = document.getElementById('advancedModal');
        this.presetSelect = document.getElementById('presetSelect');
        this.presetDescription = document.getElementById('presetDescription');
        this.previewAudio = document.getElementById('previewAudio');
        this.previewStatus = document.getElementById('previewStatus');
        
        // Buttons
        this.openBtn = document.getElementById('advancedOptionsBtn');
        this.closeBtn = document.getElementById('modalClose');
        this.cancelBtn = document.getElementById('cancelMasteringBtn');
        this.acceptBtn = document.getElementById('acceptMasteringBtn');
        this.applyPresetBtn = document.getElementById('applyPresetBtn');
        this.applyCustomEQBtn = document.getElementById('applyCustomEQBtn');
        this.resetEQBtn = document.getElementById('resetEQBtn');
        
        // EQ sliders
        this.lowShelfGain = document.getElementById('lowShelfGain');
        this.lowMidGain = document.getElementById('lowMidGain');
        this.midGain = document.getElementById('midGain');
        this.highMidGain = document.getElementById('highMidGain');
        this.highShelfGain = document.getElementById('highShelfGain');
        
        // Dynamics controls
        this.enableCompression = document.getElementById('enableCompression');
        this.enableLimiting = document.getElementById('enableLimiting');
        this.compressionControls = document.getElementById('compressionControls');
        this.limitingControls = document.getElementById('limitingControls');
        
        this.compThreshold = document.getElementById('compThreshold');
        this.compRatio = document.getElementById('compRatio');
        this.limiterCeiling = document.getElementById('limiterCeiling');
        
        // State
        this.selectedClips = [];
        this.previewPath = null;
        this.currentPreset = null;
        this.presets = [];
        
        this.setupEventListeners();
        this.loadPresets();
    }
    
    setupEventListeners() {
        // Modal controls
        this.openBtn.addEventListener('click', () => this.open());
        this.closeBtn.addEventListener('click', () => this.close());
        this.cancelBtn.addEventListener('click', () => this.close());
        this.acceptBtn.addEventListener('click', () => this.accept());
        
        // Click outside modal to close
        window.addEventListener('click', (e) => {
            if (e.target === this.modal) this.close();
        });
        
        // Preset selection
        this.presetSelect.addEventListener('change', () => this.onPresetChange());
        this.applyPresetBtn.addEventListener('click', () => this.applyPresetPreview());
        
        // Custom EQ
        this.applyCustomEQBtn.addEventListener('click', () => this.applyCustomEQPreview());
        this.resetEQBtn.addEventListener('click', () => this.resetEQ());
        
        // EQ sliders - real-time value display
        [this.lowShelfGain, this.lowMidGain, this.midGain, this.highMidGain, this.highShelfGain].forEach((slider, index) => {
            const valueSpan = document.getElementById(slider.id + 'Value');
            slider.addEventListener('input', () => {
                valueSpan.textContent = `${parseFloat(slider.value).toFixed(1)} dB`;
            });
        });
        
        // Dynamics toggles
        this.enableCompression.addEventListener('change', () => {
            this.compressionControls.style.display = this.enableCompression.checked ? 'block' : 'none';
        });
        
        this.enableLimiting.addEventListener('change', () => {
            this.limitingControls.style.display = this.enableLimiting.checked ? 'block' : 'none';
        });
        
        // Dynamics value displays
        this.compThreshold.addEventListener('input', () => {
            document.getElementById('compThresholdValue').textContent = `${this.compThreshold.value} dB`;
        });
        
        this.compRatio.addEventListener('input', () => {
            document.getElementById('compRatioValue').textContent = `${parseFloat(this.compRatio.value).toFixed(1)}:1`;
        });
        
        this.limiterCeiling.addEventListener('input', () => {
            document.getElementById('limiterCeilingValue').textContent = `${parseFloat(this.limiterCeiling.value).toFixed(1)} dB`;
        });
    }
    
    async loadPresets() {
        try {
            const response = await fetch(`${API_BASE_URL}/mastering/presets`);
            if (!response.ok) throw new Error('Failed to load presets');
            
            const data = await response.json();
            this.presets = data.presets;
            
            // Populate preset select
            this.presetSelect.innerHTML = '<option value="">-- Select Preset --</option>';
            this.presets.forEach(preset => {
                const option = document.createElement('option');
                option.value = preset.id;
                option.textContent = preset.name;
                option.dataset.description = preset.description;
                this.presetSelect.appendChild(option);
            });
            
        } catch (error) {
            console.error('Error loading presets:', error);
            showStatus('Error loading mastering presets', 'error');
        }
    }
    
    onPresetChange() {
        const selectedOption = this.presetSelect.selectedOptions[0];
        if (selectedOption && selectedOption.value) {
            this.currentPreset = selectedOption.value;
            this.presetDescription.textContent = selectedOption.dataset.description;
            this.applyPresetBtn.disabled = false;
        } else {
            this.currentPreset = null;
            this.presetDescription.textContent = 'Select a preset to see description';
            this.applyPresetBtn.disabled = true;
        }
    }
    
    open() {
        // Get selected clips from timeline
        this.selectedClips = dawTimeline.timeline.filter(clip => clip.clip_id);
        
        if (this.selectedClips.length === 0) {
            showStatus('No clips available for mastering', 'error');
            return;
        }
        
        this.modal.style.display = 'block';
        this.resetPreview();
    }
    
    close() {
        this.modal.style.display = 'none';
        this.resetPreview();
    }
    
    resetPreview() {
        this.previewPath = null;
        this.currentPreset = null;
        this.previewStatus.textContent = 'No preview loaded. Apply a preset or custom EQ to hear changes.';
        this.previewAudio.src = '';
        this.acceptBtn.disabled = true;
    }
    
    resetEQ() {
        this.lowShelfGain.value = 0;
        this.lowMidGain.value = 0;
        this.midGain.value = 0;
        this.highMidGain.value = 0;
        this.highShelfGain.value = 0;
        
        // Update displays
        document.getElementById('lowShelfValue').textContent = '0.0 dB';
        document.getElementById('lowMidValue').textContent = '0.0 dB';
        document.getElementById('midValue').textContent = '0.0 dB';
        document.getElementById('highMidValue').textContent = '0.0 dB';
        document.getElementById('highShelfValue').textContent = '0.0 dB';
        
        // Reset dynamics
        this.enableCompression.checked = false;
        this.enableLimiting.checked = true;
        this.compressionControls.style.display = 'none';
        this.limitingControls.style.display = 'block';
        
        this.compThreshold.value = -12;
        this.compRatio.value = 2;
        this.limiterCeiling.value = -0.5;
        
        document.getElementById('compThresholdValue').textContent = '-12 dB';
        document.getElementById('compRatioValue').textContent = '2.0:1';
        document.getElementById('limiterCeilingValue').textContent = '-0.5 dB';
    }
    
    async applyPresetPreview() {
        if (!this.currentPreset || this.selectedClips.length === 0) return;
        
        try {
            this.applyPresetBtn.disabled = true;
            this.applyPresetBtn.textContent = '⏳ Processing...';
            this.previewStatus.textContent = 'Generating preview...';
            
            // Use first clip for preview
            const clip = this.selectedClips[0];
            const audioPath = clip.music_path || clip.mixed_path;
            
            const response = await fetch(`${API_BASE_URL}/mastering/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    clip_id: clip.clip_id,
                    audio_path: audioPath,
                    preset: this.currentPreset
                })
            });
            
            if (!response.ok) throw new Error('Failed to generate preview');
            
            const data = await response.json();
            this.previewPath = data.preview_path;
            
            // Load preview audio
            this.previewAudio.src = this.previewPath;
            this.previewStatus.textContent = `Preview ready: ${this.presets.find(p => p.id === this.currentPreset)?.name}`;
            this.acceptBtn.disabled = false;
            
        } catch (error) {
            console.error('Error generating preview:', error);
            this.previewStatus.textContent = 'Error generating preview';
            showStatus('Failed to generate preview', 'error');
        } finally {
            this.applyPresetBtn.disabled = false;
            this.applyPresetBtn.textContent = 'Apply Preset to Preview';
        }
    }
    
    async applyCustomEQPreview() {
        if (this.selectedClips.length === 0) return;
        
        try {
            this.applyCustomEQBtn.disabled = true;
            this.applyCustomEQBtn.textContent = '⏳ Processing...';
            this.previewStatus.textContent = 'Generating custom EQ preview...';
            
            // Build EQ bands
            const eq_bands = [
                { type: 'lowshelf', frequency: 100, gain: parseFloat(this.lowShelfGain.value), q: 0.7 },
                { type: 'peak', frequency: 500, gain: parseFloat(this.lowMidGain.value), q: 1.0 },
                { type: 'peak', frequency: 2000, gain: parseFloat(this.midGain.value), q: 1.0 },
                { type: 'peak', frequency: 5000, gain: parseFloat(this.highMidGain.value), q: 1.0 },
                { type: 'highshelf', frequency: 10000, gain: parseFloat(this.highShelfGain.value), q: 0.7 }
            ];
            
            // Build dynamics settings
            const compression = this.enableCompression.checked ? {
                threshold: parseFloat(this.compThreshold.value),
                ratio: parseFloat(this.compRatio.value),
                attack: 5,
                release: 100
            } : null;
            
            const limiting = this.enableLimiting.checked ? {
                threshold: parseFloat(this.limiterCeiling.value),
                release: 100
            } : null;
            
            // Use first clip for preview
            const clip = this.selectedClips[0];
            const audioPath = clip.music_path || clip.mixed_path;
            
            const response = await fetch(`${API_BASE_URL}/mastering/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    clip_id: clip.clip_id,
                    audio_path: audioPath,
                    eq_bands,
                    compression,
                    limiting
                })
            });
            
            if (!response.ok) throw new Error('Failed to generate preview');
            
            const data = await response.json();
            this.previewPath = data.preview_path;
            
            // Load preview audio
            this.previewAudio.src = this.previewPath;
            this.previewStatus.textContent = 'Custom EQ preview ready';
            this.acceptBtn.disabled = false;
            this.currentPreset = null; // Clear preset since using custom
            
        } catch (error) {
            console.error('Error generating preview:', error);
            this.previewStatus.textContent = 'Error generating preview';
            showStatus('Failed to generate custom EQ preview', 'error');
        } finally {
            this.applyCustomEQBtn.disabled = false;
            this.applyCustomEQBtn.textContent = 'Apply Custom EQ to Preview';
        }
    }
    
    async accept() {
        if (!this.previewPath) {
            showStatus('No preview to apply', 'error');
            return;
        }
        
        try {
            this.acceptBtn.disabled = true;
            this.acceptBtn.textContent = '⏳ Applying...';
            
            // Apply to all selected clips
            for (const clip of this.selectedClips) {
                const audioPath = clip.music_path || clip.mixed_path;
                
                let response;
                if (this.currentPreset) {
                    // Apply preset
                    response = await fetch(`${API_BASE_URL}/mastering/apply-preset`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            clip_id: clip.clip_id,
                            audio_path: audioPath,
                            preset: this.currentPreset
                        })
                    });
                } else {
                    // Apply custom EQ
                    const eq_bands = [
                        { type: 'lowshelf', frequency: 100, gain: parseFloat(this.lowShelfGain.value), q: 0.7 },
                        { type: 'peak', frequency: 500, gain: parseFloat(this.lowMidGain.value), q: 1.0 },
                        { type: 'peak', frequency: 2000, gain: parseFloat(this.midGain.value), q: 1.0 },
                        { type: 'peak', frequency: 5000, gain: parseFloat(this.highMidGain.value), q: 1.0 },
                        { type: 'highshelf', frequency: 10000, gain: parseFloat(this.highShelfGain.value), q: 0.7 }
                    ];
                    
                    const compression = this.enableCompression.checked ? {
                        threshold: parseFloat(this.compThreshold.value),
                        ratio: parseFloat(this.compRatio.value),
                        attack: 5,
                        release: 100
                    } : null;
                    
                    const limiting = this.enableLimiting.checked ? {
                        threshold: parseFloat(this.limiterCeiling.value),
                        release: 100
                    } : null;
                    
                    response = await fetch(`${API_BASE_URL}/mastering/apply-custom-eq`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            clip_id: clip.clip_id,
                            audio_path: audioPath,
                            eq_bands,
                            compression,
                            limiting
                        })
                    });
                }
                
                if (!response.ok) throw new Error('Failed to apply mastering');
                
                const data = await response.json();
                
                // Update clip path in timeline
                if (clip.music_path) {
                    clip.music_path = data.processed_path;
                } else if (clip.mixed_path) {
                    clip.mixed_path = data.processed_path;
                }
            }
            
            showStatus(`✅ Mastering applied to ${this.selectedClips.length} clip(s)`, 'success');
            
            // Reload timeline
            if (window.loadTimeline) {
                loadTimeline();
            }
            
            this.close();
            
        } catch (error) {
            console.error('Error applying mastering:', error);
            showStatus('Failed to apply mastering', 'error');
        } finally {
            this.acceptBtn.disabled = false;
            this.acceptBtn.textContent = 'Accept & Apply to Timeline';
        }
    }
}

// Initialize when DOM is ready
let masteringController;
document.addEventListener('DOMContentLoaded', () => {
    if (typeof API_BASE_URL !== 'undefined') {
        masteringController = new MasteringController();
    }
});
