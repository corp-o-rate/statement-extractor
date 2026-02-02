'use client';

import Link from 'next/link';
import { Github, ExternalLink, Package, Container } from 'lucide-react';

export function Header() {
  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-black tracking-tight">
                STATEMENT EXTRACTOR
              </span>
            </Link>
            <span className="hidden sm:inline text-xs text-gray-400 border-l pl-3 ml-1">
              by{' '}
              <a
                href="https://corp-o-rate.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-red-600 hover:underline"
              >
                corp-o-rate
              </a>
            </span>
          </div>

          {/* Navigation */}
          <nav className="flex items-center gap-4">
            <Link
              href="/docs"
              className="hidden sm:inline text-sm text-gray-600 hover:text-black transition-colors font-medium"
            >
              Docs
            </Link>
            <a
              href="#llm-prompts"
              className="hidden sm:inline text-sm text-gray-600 hover:text-black transition-colors"
            >
              AI Prompts
            </a>
            <a
              href="https://pypi.org/project/corp-extractor/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on PyPI"
            >
              <Package className="w-5 h-5" />
            </a>
            <a
              href="https://hub.docker.com/r/corporate/statement-extractor"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on Docker Hub"
            >
              <Container className="w-5 h-5" />
            </a>
            <a
              href="https://huggingface.co/Corp-o-Rate-Community"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on Hugging Face"
            >
              <svg className="w-5 h-5" viewBox="0 0 32 32" fill="currentColor">
                <path d="M16 0C7.163 0 0 7.163 0 16s7.163 16 16 16 16-7.163 16-16S24.837 0 16 0zm-4.5 22.5c-1.933 0-3.5-1.567-3.5-3.5s1.567-3.5 3.5-3.5 3.5 1.567 3.5 3.5-1.567 3.5-3.5 3.5zm9 0c-1.933 0-3.5-1.567-3.5-3.5s1.567-3.5 3.5-3.5 3.5 1.567 3.5 3.5-1.567 3.5-3.5 3.5zM10 13c-.828 0-1.5-.672-1.5-1.5v-2c0-.828.672-1.5 1.5-1.5s1.5.672 1.5 1.5v2c0 .828-.672 1.5-1.5 1.5zm12 0c-.828 0-1.5-.672-1.5-1.5v-2c0-.828.672-1.5 1.5-1.5s1.5.672 1.5 1.5v2c0 .828-.672 1.5-1.5 1.5z"/>
              </svg>
            </a>
            <a
              href="https://github.com/corp-o-rate/statement-extractor"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
          </nav>
        </div>
      </div>
    </header>
  );
}
