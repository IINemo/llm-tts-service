import axios from 'axios';

// Base URL for the service API
const API_BASE_URL = 'http://localhost:8001';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes timeout for long-running ToT requests
});

/**
 * Start an async ToT job
 * @param {Object} params - Request parameters
 * @param {string} params.message - The user's question/prompt
 * @param {string} params.strategy - TTS strategy (tree_of_thoughts, deepconf, etc.)
 * @param {string} params.model - Model to use (default: openai/gpt-4o-mini)
 * @param {Object} params.strategyConfig - Strategy-specific configuration
 * @returns {Promise<Object>} Job creation response with job_id
 */
export async function startJob({
  message,
  strategy = 'tree_of_thoughts',
  model = 'openai/gpt-4o-mini',
  strategyConfig = {},
}) {
  const requestBody = {
    model,
    prompt: message,
    tts_strategy: strategy,
    ...strategyConfig,
  };

  try {
    const response = await apiClient.post('/v1/jobs', requestBody);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw {
      message: error.response?.data?.error?.message || error.message,
      status: error.response?.status,
      details: error.response?.data,
    };
  }
}

/**
 * Poll job status
 * @param {string} jobId - Job ID to check
 * @returns {Promise<Object>} Job status with progress and results
 */
export async function pollJobStatus(jobId) {
  try {
    const response = await apiClient.get(`/v1/jobs/${jobId}`);
    return response.data;
  } catch (error) {
    console.error('Polling Error:', error);
    throw {
      message: error.response?.data?.error?.message || error.message,
      status: error.response?.status,
      details: error.response?.data,
    };
  }
}

/**
 * Start job and poll until completion (convenience function)
 * @param {Object} params - Same as startJob
 * @param {Function} onProgress - Callback for progress updates (optional)
 * @param {number} pollInterval - Polling interval in ms (default: 1000)
 * @returns {Promise<Object>} Final job result
 */
export async function executeJobWithPolling(params, onProgress = null, pollInterval = 1000) {
  // Start the job
  const { job_id } = await startJob(params);

  // Poll for status
  return new Promise((resolve, reject) => {
    const pollTimer = setInterval(async () => {
      try {
        const status = await pollJobStatus(job_id);

        // Call progress callback if provided
        if (onProgress) {
          onProgress(status);
        }

        // Check if job is done
        if (status.status === 'completed') {
          clearInterval(pollTimer);
          resolve(status);
        } else if (status.status === 'failed') {
          clearInterval(pollTimer);
          reject(new Error(status.error || 'Job failed'));
        }
      } catch (error) {
        clearInterval(pollTimer);
        reject(error);
      }
    }, pollInterval);
  });
}

/**
 * Health check endpoint
 */
export async function checkHealth() {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw error;
  }
}

/**
 * Get available models
 */
export async function getModels() {
  try {
    const response = await apiClient.get('/v1/models');
    return response.data;
  } catch (error) {
    throw error;
  }
}

export default {
  startJob,
  pollJobStatus,
  executeJobWithPolling,
  checkHealth,
  getModels,
};
