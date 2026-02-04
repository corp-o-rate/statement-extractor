'use client';

import Link from 'next/link';
import Image from 'next/image';
import { Github } from 'lucide-react';

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
              <Image src="https://pypi.org/static/images/logo-small.8998e9d1.svg" alt="PyPI" width={20} height={20} className="w-5 h-5" />
            </a>
            <a
              href="https://hub.docker.com/repository/docker/neilellis/statement-extractor-runpod"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on Docker Hub"
            >
              <Image src="https://www.docker.com/wp-content/uploads/2024/02/cropped-docker-logo-favicon-32x32.png" alt="Docker" width={20} height={20} className="w-5 h-5" />
            </a>
            <a
              href="https://huggingface.co/Corp-o-Rate-Community"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on Hugging Face"
            >
              <Image src="https://huggingface.co/favicon.ico" alt="Hugging Face" width={20} height={20} className="w-5 h-5" />
            </a>
            <a
              href="https://colab.research.google.com/github/corp-o-rate/statement-extractor/blob/main/statement_extractor_demo.ipynb"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="Open in Google Colab"
            >
              <Image src="https://colab.research.google.com/favicon.ico" alt="Google Colab" width={20} height={20} className="w-5 h-5" />
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
