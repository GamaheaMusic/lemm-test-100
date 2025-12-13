// API Configuration
const API_BASE_URL = 'http://localhost:7860/api';

// State
let timeline = [];
let dawTimeline;

// DOM Elements
const promptInput = document.getElementById('prompt');
const lyricsInput = document.getElementById('lyrics');
const durationInput = document.getElementById('duration');
const generateBtn = document.getElementById('generateBtn');
const autoGenLyricsBtn = document.getElementById('autoGenLyricsBtn');
const exportBtn = document.getElementById('exportBtn');
const statusDiv = document.getElementById('status');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    dawTimeline = new DAWTimeline();
    loadTimeline();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    generateBtn.addEventListener('click', generateMusic);
    autoGenLyricsBtn.addEventListener('click', generateLyrics);
    exportBtn.addEventListener('click', exportTimeline);
}

// Generate lyrics from prompt
async function generateLyrics() {
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        showStatus('Please enter a prompt first', 'error');
        return;
    }
    
    try {
        autoGenLyricsBtn.disabled = true;
        autoGenLyricsBtn.textContent = '⏳ Generating lyrics...';
        showStatus('Analyzing prompt and generating lyrics...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/generate-lyrics`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                duration: parseInt(durationInput.value)
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate lyrics');
        }
        
        const result = await response.json();
        lyricsInput.value = result.lyrics;
        
        // Show analysis info if available
        if (result.analysis) {
            const genres = result.analysis.genres || [];
            const mood = result.analysis.mood || 'unknown';
            const genreText = genres.length > 0 ? genres.join(', ') : 'general';
            showStatus(`✅ Lyrics generated (Style: ${genreText}, Mood: ${mood})`, 'success');
        } else {
            showStatus('✅ Lyrics generated successfully', 'success');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        autoGenLyricsBtn.disabled = false;
        autoGenLyricsBtn.textContent = '🎤 Generate Lyrics';
    }
}

// Generate music clip
async function generateMusic() {
    const prompt = promptInput.value.trim();
    const lyrics = lyricsInput.value.trim();
    const duration = parseInt(durationInput.value);
    const position = document.querySelector('input[name="position"]:checked').value;
    
    if (!prompt) {
        showStatus('Please enter a music prompt', 'error');
        return;
    }
    
    const statusMsg = lyrics 
        ? 'Generating music with vocals... This may take a moment.'
        : 'Generating instrumental music... This may take a moment.';
    
    showStatus(statusMsg, 'info');
    setButtonLoading(generateBtn, true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/generation/generate-music`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                lyrics: lyrics || null,
                duration: duration
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show analysis info if available
            if (data.analysis) {
                console.log('Prompt analysis:', data.analysis);
            }
            
            // Add clip to timeline
            await addClipToTimeline({
                clip_id: data.clip_id,
                file_path: data.file_path,
                duration: data.duration,
                position: position
            });
            
            // Build success message with style consistency info
            let message = lyrics
                ? 'Music clip with vocals generated and added to timeline!'
                : 'Instrumental music clip generated and added to timeline!';
            
            if (data.style_consistent && data.num_reference_clips > 0) {
                message += ` (Style matched to ${data.num_reference_clips} existing clip${data.num_reference_clips > 1 ? 's' : ''})`;
            }
            
            showStatus(message, 'success');
            await loadTimeline();
        } else {
            throw new Error(data.error || 'Failed to generate music');
        }
    } catch (error) {
        console.error('Error generating music:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        setButtonLoading(generateBtn, false);
    }
}

// Add clip to timeline
async function addClipToTimeline(clipData) {
    const response = await fetch(`${API_BASE_URL}/timeline/clips`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(clipData)
    });
    
    const data = await response.json();
    
    if (!data.success) {
        throw new Error(data.error || 'Failed to add clip to timeline');
    }
    
    return data;
}

// Load timeline
async function loadTimeline() {
    try {
        const response = await fetch(`${API_BASE_URL}/timeline/clips`);
        const data = await response.json();
        
        if (data.success) {
            timeline = data.clips;
            dawTimeline.loadTimeline(timeline);
            exportBtn.disabled = timeline.length === 0;
        }
    } catch (error) {
        console.error('Error loading timeline:', error);
    }
}

// Remove clip
async function removeClip(clipId) {
    if (!confirm('Are you sure you want to remove this clip?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/timeline/clips/${clipId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Clip removed', 'success');
            await loadTimeline();
        } else {
            throw new Error(data.error || 'Failed to remove clip');
        }
    } catch (error) {
        console.error('Error removing clip:', error);
        showStatus(`Error: ${error.message}`, 'error');
    }
}

// Export timeline
async function exportTimeline() {
    const filename = prompt('Enter filename (without extension):', 'my_song');
    
    if (!filename) {
        return;
    }
    
    showStatus('Exporting timeline...', 'info');
    setButtonLoading(exportBtn, true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/export/merge`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: filename,
                format: 'wav'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Download the file
            const downloadUrl = `${API_BASE_URL}/export/download/${data.filename}`;
            window.open(downloadUrl, '_blank');
            showStatus('Export successful! Download started.', 'success');
        } else {
            throw new Error(data.error || 'Failed to export timeline');
        }
    } catch (error) {
        console.error('Error exporting timeline:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        setButtonLoading(exportBtn, false);
    }
}

// Clear timeline
async function clearTimeline() {
    if (!confirm('Are you sure you want to clear the entire timeline?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/timeline/clear`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Timeline cleared', 'success');
            await loadTimeline();
        } else {
            throw new Error(data.error || 'Failed to clear timeline');
        }
    } catch (error) {
        console.error('Error clearing timeline:', error);
        showStatus(`Error: ${error.message}`, 'error');
    }
}

// Expose removeClip globally for onclick handlers
window.removeClip = removeClip;

// Utility functions
function showStatus(message, type = 'info') {
    statusDiv.textContent = message;
    statusDiv.className = `status-message show ${type}`;
    
    setTimeout(() => {
        statusDiv.classList.remove('show');
    }, 5000);
}

function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.innerHTML = '<span class="loading"></span>Loading...';
    } else {
        button.disabled = false;
        button.textContent = button.dataset.originalText;
    }
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
