'use client';

import { Clock, Zap, Server, Heart, X, ExternalLink } from 'lucide-react';

interface WarmUpDialogProps {
  isOpen: boolean;
  elapsedSeconds: number;
  onClose: () => void;
  isTimeout?: boolean;
}

export function WarmUpDialog({ isOpen, elapsedSeconds, onClose, isTimeout }: WarmUpDialogProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog */}
      <div className="relative bg-white border-2 border-black shadow-[6px_6px_0_0_#000] max-w-lg w-full animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b-2 border-black">
          <div className="flex items-center gap-3">
            {isTimeout ? (
              <div className="w-10 h-10 bg-orange-100 flex items-center justify-center">
                <Clock className="w-5 h-5 text-orange-600" />
              </div>
            ) : (
              <div className="w-10 h-10 bg-blue-100 flex items-center justify-center">
                <Server className="w-5 h-5 text-blue-600 spinner" />
              </div>
            )}
            <div>
              <h3 className="font-bold text-lg">
                {isTimeout ? 'Request Timed Out' : 'Starting GPU Instance'}
              </h3>
              <p className="text-sm text-gray-500">
                {isTimeout ? 'The server is still warming up' : `Processing... ${elapsedSeconds}s`}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {isTimeout ? (
            <>
              <p className="text-gray-700">
                The GPU server is still warming up. This happens when there hasn&apos;t been a request
                in a while and a new instance needs to start.
              </p>
              <p className="text-gray-700">
                <strong>Please try again</strong> — the server should be ready now or very soon.
                Subsequent requests will be much faster (typically 5-15 seconds).
              </p>
            </>
          ) : (
            <>
              <p className="text-gray-700">
                We&apos;re starting a fresh GPU instance just for you. This cold start typically takes
                about <strong>60 seconds</strong> on the first request.
              </p>
              <div className="flex items-start gap-3 p-3 bg-green-50 border border-green-200">
                <Zap className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-green-800">
                  <strong>Good news:</strong>The library can be run locally and it goes much faster, or reach out to us to pay to keep a GPU warm 24/7.
                </p>
              </div>
            </>
          )}

          <div className="pt-2 space-y-3">
            <p className="text-sm font-semibold text-gray-900">Want faster results?</p>
            <div className="grid gap-2">
              <a
                href="/docs"
                className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-200 hover:border-black hover:bg-gray-100 transition-all group"
              >
                <Server className="w-4 h-4 text-gray-600" />
                <span className="font-medium">Run locally</span>
                <span className="text-sm text-gray-500 ml-auto">Instant results, no wait</span>
                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-black" />
              </a>
              <a
                href="mailto:neil@corp-o-rate.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 hover:border-red-600 hover:bg-red-100 transition-all group"
              >
                <Heart className="w-4 h-4 text-red-600" />
                <span className="font-medium text-red-700">Support the project</span>
                <span className="text-sm text-red-600 ml-auto">Help us keep a GPU warm 24/7</span>
                <ExternalLink className="w-4 h-4 text-red-400 group-hover:text-red-600" />
              </a>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-gray-50 border-t border-gray-200">
          <button
            onClick={onClose}
            className="w-full btn-primary"
          >
            {isTimeout ? 'Try Again' : 'Got it, keep waiting'}
          </button>
        </div>
      </div>
    </div>
  );
}
