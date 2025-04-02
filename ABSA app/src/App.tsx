import React, { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';
import axios from 'axios';
import type { AnalysisResponse } from './types';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/analyze', { text });
      console.log(response)
      setResult(response.data);
    } catch (err) {
      setError('Failed to analyze text. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          Aspect-Based Sentiment Analysis
        </h1>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-2">
              Enter your text for analysis
            </label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full h-32 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="The food was delicious but the service was slow..."
            />
          </div>
          
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="-ml-1 mr-2 h-5 w-5" />
                Analyze
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h2>
            <div className="bg-white shadow rounded-lg overflow-hidden">
            {
                    JSON.stringify(result, null, 2)
                  }
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;