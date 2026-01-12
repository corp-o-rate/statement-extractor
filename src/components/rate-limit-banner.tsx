'use client';

import { AlertTriangle, ExternalLink, Heart } from 'lucide-react';

interface RateLimitBannerProps {
  message?: string;
}

export function RateLimitBanner({ message }: RateLimitBannerProps) {
  if (!message) return null;

  return (
    <div className="rate-limit-banner mb-6">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-semibold text-gray-900">{message}</p>
          <div className="mt-3 flex flex-wrap gap-4 text-sm">
            <a
              href="#run-locally"
              className="inline-flex items-center gap-1 text-red-600 hover:underline"
            >
              Run locally for unlimited usage
              <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="https://github.com/sponsors/neilellis"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-gray-600 hover:text-gray-900"
            >
              <Heart className="w-3 h-3" />
              Support the project
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
