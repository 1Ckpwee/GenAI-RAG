/**
 * RAG Document Search - Client-side JavaScript
 */

// Function to load documents list
function loadDocuments() {
    fetch('/documents')
        .then(response => response.json())
        .then(data => {
            const documentsList = document.getElementById('documentsList');
            if (data.documents.length === 0) {
                documentsList.innerHTML = '<p>No documents uploaded yet.</p>';
                return;
            }

            let html = '<ul class="list-group">';
            data.documents.forEach(doc => {
                const date = new Date(doc.uploaded_at * 1000).toLocaleString();
                const size = (doc.size / 1024).toFixed(2) + ' KB';
                html += `
                    <li class="list-group-item">
                        <div class="d-flex w-100 justify-content-between">
                            <h5 class="mb-1">${doc.filename}</h5>
                            <small>${date}</small>
                        </div>
                        <p class="mb-1">Size: ${size}</p>
                    </li>
                `;
            });
            html += '</ul>';
            documentsList.innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading documents:', error);
            document.getElementById('documentsList').innerHTML = '<p>Error loading documents.</p>';
        });
}

// Handle file upload
function handleFileUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show spinner and progress
    document.getElementById('uploadSpinner').style.display = 'inline-block';
    const progressBar = document.getElementById('uploadProgress');
    progressBar.style.display = 'flex';
    progressBar.querySelector('.progress-bar').style.width = '0%';
    
    // Clear previous results
    document.getElementById('uploadResult').innerHTML = '';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // Update progress
        progressBar.querySelector('.progress-bar').style.width = '100%';
        
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Upload failed');
            });
        }
        return response.json();
    })
    .then(data => {
        // Hide spinner
        document.getElementById('uploadSpinner').style.display = 'none';
        
        // Show success message
        document.getElementById('uploadResult').innerHTML = `
            <div class="alert alert-success">
                File "${data.filename}" uploaded successfully.
                Processed ${data.document_count} documents into ${data.chunks_count} chunks.
            </div>
        `;
        
        // Reset file input
        fileInput.value = '';
        
        // Reload documents list
        loadDocuments();
    })
    .catch(error => {
        // Hide spinner
        document.getElementById('uploadSpinner').style.display = 'none';
        
        // Show error message
        document.getElementById('uploadResult').innerHTML = `
            <div class="alert alert-danger">
                ${error.message}
            </div>
        `;
    });
}

// Handle query submission
function handleQuerySubmission(event) {
    event.preventDefault();
    
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a question.');
        return;
    }
    
    // Show spinner and progress
    document.getElementById('querySpinner').style.display = 'inline-block';
    const progressBar = document.getElementById('queryProgress');
    progressBar.style.display = 'flex';
    progressBar.querySelector('.progress-bar').style.width = '0%';
    
    // Clear previous results
    document.getElementById('answer').innerHTML = '';
    document.getElementById('sourcesList').innerHTML = '';
    
    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => {
        // Update progress
        progressBar.querySelector('.progress-bar').style.width = '100%';
        
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Query failed');
            });
        }
        return response.json();
    })
    .then(data => {
        // Hide spinner
        document.getElementById('querySpinner').style.display = 'none';
        
        // Show answer
        document.getElementById('answer').innerHTML = data.answer;
        
        // Show sources
        const sourcesList = document.getElementById('sourcesList');
        if (data.sources.length === 0) {
            sourcesList.innerHTML = '<p>No sources found.</p>';
            return;
        }
        
        let html = '';
        data.sources.forEach((source, index) => {
            html += `
                <div class="source-item">
                    <strong>Source ${index + 1}: ${source.source}</strong>
                    <p>${source.content}</p>
                </div>
            `;
        });
        sourcesList.innerHTML = html;
    })
    .catch(error => {
        // Hide spinner
        document.getElementById('querySpinner').style.display = 'none';
        
        // Show error message
        document.getElementById('answer').innerHTML = `
            <div class="alert alert-danger">
                ${error.message}
            </div>
        `;
    });
}

// Initialize the application
function initApp() {
    // Load documents on page load
    loadDocuments();
    
    // Set up event listeners
    document.getElementById('uploadForm').addEventListener('submit', handleFileUpload);
    document.getElementById('queryForm').addEventListener('submit', handleQuerySubmission);
}

// Run initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp); 