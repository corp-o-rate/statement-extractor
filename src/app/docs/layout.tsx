import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Documentation | Statement Extractor',
  description: 'Complete documentation for the Statement Extractor Python library and T5-Gemma 2 model. Learn how to extract subject-predicate-object triples from text.',
  openGraph: {
    title: 'Statement Extractor Documentation',
    description: 'Complete documentation for the Statement Extractor Python library and T5-Gemma 2 model.',
    type: 'website',
  },
};

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-white">
      {children}
    </div>
  );
}
