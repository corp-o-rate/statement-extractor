'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, Tags } from 'lucide-react';

// Canonical predicates organized by category
const PREDICATE_CATEGORIES = [
  {
    name: 'Ownership & Corporate Structure',
    predicates: [
      'Owned By', 'Parent Organization', 'Ultimate Parent', 'Has Subsidiary',
      'Acquired', 'Merged With', 'Spun Off From', 'Controlling Shareholder',
      'Minority Shareholder', 'Beneficial Owner', 'Person of Significant Control',
      'Nominee For', 'Succeeded By', 'Preceded By',
    ],
  },
  {
    name: 'Investment & Finance',
    predicates: [
      'Investor', 'Funded By', 'Creditor', 'Debtor', 'VC Backed By', 'PE Backed By',
      'IPO Underwriter', 'Bond Holder',
    ],
  },
  {
    name: 'Leadership & Employment',
    predicates: [
      'Chief Executive Officer', 'Chief Financial Officer', 'Chief Operating Officer',
      'Founder', 'Board Member', 'Chairperson', 'Company Secretary', 'Employee',
      'Former Employee', 'Former Director', 'Advisor', 'Consultant',
    ],
  },
  {
    name: 'Organizational Structure',
    predicates: ['Division Of', 'Department Of'],
  },
  {
    name: 'Supply Chain',
    predicates: [
      'Supplier', 'Customer', 'Manufacturer', 'Distributor', 'Contractor',
      'Outsources To', 'Subcontractor', 'Raw Material Source',
    ],
  },
  {
    name: 'Geography & Jurisdiction',
    predicates: [
      'Headquarters', 'Located In', 'Operates In', 'Facility In', 'Registered In',
      'Tax Residence', 'Offshore Entity In', 'Branch In', 'Citizenship', 'Formed In', 'Residence',
    ],
  },
  {
    name: 'Legal & Regulatory',
    predicates: [
      'Sued', 'Sued By', 'Fined By', 'Regulated By', 'Licensed By', 'Sanctioned By',
      'Investigated By', 'Settled With', 'Consent Decree With', 'Debarred By',
    ],
  },
  {
    name: 'Political',
    predicates: [
      'Lobbies', 'Donated To', 'Endorsed By', 'Member Of', 'Sponsored By',
      'Lobbied By', 'PAC Contribution', 'Revolving Door',
    ],
  },
  {
    name: 'Environmental & Social',
    predicates: [
      'Polluted', 'Affected Community', 'Displaced', 'Deforested', 'Benefited',
      'Restored', 'Employed In', 'Invested In Community', 'Violated Rights',
      'Emitted GHG', 'Water Usage', 'Waste Disposal',
    ],
  },
  {
    name: 'Products & IP',
    predicates: [
      'Brand Of', 'Product Of', 'Trademark Of', 'Licensed From', 'White Label For',
      'Recalls', 'Developer', 'Publisher',
    ],
  },
  {
    name: 'Business Relationships',
    predicates: [
      'Partner', 'Joint Venture With', 'Franchisee Of', 'Distributor For',
      'Licensed To', 'Exclusive Dealer', 'Operator',
    ],
  },
  {
    name: 'Personal Relationships',
    predicates: ['Spouse', 'Relative', 'Associate', 'Co-Founder', 'Classmate', 'Club Member'],
  },
  {
    name: 'Classification',
    predicates: ['Industry', 'Competitor', 'Similar To', 'Same Sector', 'Peer Of', 'Instance Of'],
  },
  {
    name: 'Events & Mentions',
    predicates: [
      'Mentioned With', 'Accused Of', 'Praised For', 'Criticized For',
      'Announced', 'Rumored', 'Participant',
    ],
  },
];

export function CanonicalPredicates() {
  const [isExpanded, setIsExpanded] = useState(false);

  const totalPredicates = PREDICATE_CATEGORIES.reduce(
    (sum, cat) => sum + cat.predicates.length,
    0
  );

  return (
    <div id="canonical-predicates" className="scroll-mt-24">
      <div className="editorial-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Tags className="w-5 h-5 text-red-600" />
            <h3 className="font-bold text-lg">Canonical Predicates</h3>
            <span className="text-sm text-gray-500">({totalPredicates} predicates)</span>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-black transition-colors"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-4 h-4" />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                Expand
              </>
            )}
          </button>
        </div>

        <p className="text-gray-600 mb-4">
          When &quot;Use canonical predicates&quot; is enabled, extracted predicates are normalized to
          these standard forms using embedding similarity. This helps create consistent knowledge
          graphs and enables better analysis across documents.
        </p>

        {isExpanded ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {PREDICATE_CATEGORIES.map((category) => (
              <div key={category.name} className="bg-gray-50 p-3 rounded border">
                <h4 className="font-semibold text-sm text-gray-700 mb-2">
                  {category.name}
                </h4>
                <div className="flex flex-wrap gap-1">
                  {category.predicates.map((pred) => (
                    <span
                      key={pred}
                      className="text-xs px-2 py-0.5 bg-white border border-gray-200 rounded text-gray-600"
                    >
                      {pred}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {PREDICATE_CATEGORIES.slice(0, 3).flatMap((cat) => cat.predicates.slice(0, 4)).map((pred) => (
              <span
                key={pred}
                className="text-xs px-2 py-0.5 bg-gray-100 border border-gray-200 rounded text-gray-600"
              >
                {pred}
              </span>
            ))}
            <span className="text-xs px-2 py-0.5 text-gray-400">
              ...and {totalPredicates - 12} more
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
