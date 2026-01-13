/**
 * Parse XML statements from T5-Gemma 2 model output
 *
 * Expected format:
 * <statements>
 *   <stmt>
 *     <subject type="ORG">Apple Inc.</subject>
 *     <object type="EVENT">carbon neutral by 2030</object>
 *     <predicate>committed to</predicate>
 *     <text>Apple Inc. committed to becoming carbon neutral by 2030.</text>
 *   </stmt>
 * </statements>
 */

import { Statement, EntityType, Entity } from './types';

/**
 * Parse entity type from string, with fallback to UNKNOWN
 */
function parseEntityType(typeStr: string | null): EntityType {
  if (!typeStr) return 'UNKNOWN';

  const normalized = typeStr.toUpperCase().trim();
  const validTypes: EntityType[] = [
    'ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
    'WORK_OF_ART', 'LAW', 'DATE', 'MONEY', 'PERCENT', 'QUANTITY'
  ];

  if (validTypes.includes(normalized as EntityType)) {
    return normalized as EntityType;
  }

  return 'UNKNOWN';
}

/**
 * Extract text content and type from an entity element
 */
function parseEntity(element: Element | null): Entity {
  if (!element) {
    return { name: '', type: 'UNKNOWN' };
  }

  const type = parseEntityType(element.getAttribute('type'));
  const name = element.textContent?.trim() || '';

  return { name, type };
}

/**
 * Parse a single statement element
 */
function parseStatementElement(stmtElement: Element): Statement | null {
  const subjectEl = stmtElement.querySelector('subject');
  const objectEl = stmtElement.querySelector('object');
  const predicateEl = stmtElement.querySelector('predicate');
  const textEl = stmtElement.querySelector('text');

  const subject = parseEntity(subjectEl);
  const object = parseEntity(objectEl);
  const predicate = predicateEl?.textContent?.trim() || '';
  const text = textEl?.textContent?.trim() || '';

  // Skip statements with missing required fields
  if (!subject.name || !predicate) {
    return null;
  }

  return {
    subject,
    object,
    predicate,
    text: text || `${subject.name} ${predicate} ${object.name}`.trim(),
  };
}

/**
 * Create a normalized key for deduplication
 * Uses lowercase trimmed subject name + predicate + object name
 */
function getStatementKey(statement: Statement): string {
  const subject = statement.subject.name.toLowerCase().trim();
  const predicate = statement.predicate.toLowerCase().trim();
  const object = statement.object.name.toLowerCase().trim();
  return `${subject}|||${predicate}|||${object}`;
}

/**
 * Remove duplicate statements based on subject-predicate-object triples
 * Keeps the first occurrence of each unique triple
 */
function deduplicateStatements(statements: Statement[]): Statement[] {
  const seen = new Set<string>();
  const unique: Statement[] = [];

  for (const statement of statements) {
    const key = getStatementKey(statement);
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(statement);
    }
  }

  return unique;
}

/**
 * Parse XML string containing statements
 */
export function parseStatements(xmlString: string): Statement[] {
  const statements: Statement[] = [];

  // Handle empty or invalid input
  if (!xmlString || typeof xmlString !== 'string') {
    return statements;
  }

  // Trim and check for valid XML structure
  const trimmed = xmlString.trim();
  if (!trimmed.includes('<statements>') && !trimmed.includes('<stmt>')) {
    console.warn('No <statements> or <stmt> tags found in output');
    return statements;
  }

  try {
    // Wrap in root if needed
    let xmlToParse = trimmed;
    if (!trimmed.startsWith('<statements>')) {
      xmlToParse = `<statements>${trimmed}</statements>`;
    }

    // Parse using DOMParser (works in browser and Node with jsdom)
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlToParse, 'text/xml');

    // Check for parse errors
    const parseError = doc.querySelector('parsererror');
    if (parseError) {
      console.error('XML parse error:', parseError.textContent);
      // Try fallback regex-based parsing
      return deduplicateStatements(parseStatementsRegex(trimmed));
    }

    // Extract all statement elements
    const stmtElements = doc.querySelectorAll('stmt');

    for (const stmtEl of stmtElements) {
      const statement = parseStatementElement(stmtEl);
      if (statement) {
        statements.push(statement);
      }
    }
  } catch (error) {
    console.error('Error parsing XML statements:', error);
    // Try fallback regex-based parsing
    return deduplicateStatements(parseStatementsRegex(trimmed));
  }

  // Deduplicate before returning
  return deduplicateStatements(statements);
}

/**
 * Fallback regex-based parser for malformed XML
 */
function parseStatementsRegex(xmlString: string): Statement[] {
  const statements: Statement[] = [];

  // Match individual statements
  const stmtRegex = /<stmt>([\s\S]*?)<\/stmt>/g;
  let match;

  while ((match = stmtRegex.exec(xmlString)) !== null) {
    const stmtContent = match[1];

    // Extract fields
    const subjectMatch = stmtContent.match(/<subject(?:\s+type="([^"]*)")?>([\s\S]*?)<\/subject>/);
    const objectMatch = stmtContent.match(/<object(?:\s+type="([^"]*)")?>([\s\S]*?)<\/object>/);
    const predicateMatch = stmtContent.match(/<predicate>([\s\S]*?)<\/predicate>/);
    const textMatch = stmtContent.match(/<text>([\s\S]*?)<\/text>/);

    const subject: Entity = {
      name: subjectMatch?.[2]?.trim() || '',
      type: parseEntityType(subjectMatch?.[1] || null),
    };

    const object: Entity = {
      name: objectMatch?.[2]?.trim() || '',
      type: parseEntityType(objectMatch?.[1] || null),
    };

    const predicate = predicateMatch?.[1]?.trim() || '';
    const text = textMatch?.[1]?.trim() || '';

    if (subject.name && predicate) {
      statements.push({
        subject,
        object,
        predicate,
        text: text || `${subject.name} ${predicate} ${object.name}`.trim(),
      });
    }
  }

  return statements;
}

/**
 * Convert statements to graph data for visualization
 */
export function statementsToGraphData(statements: Statement[]) {
  const nodesMap = new Map<string, { name: string; type: EntityType }>();
  const links: Array<{ source: string; target: string; predicate: string }> = [];

  for (const stmt of statements) {
    // Add subject node
    const subjectId = `${stmt.subject.type}:${stmt.subject.name}`;
    if (!nodesMap.has(subjectId)) {
      nodesMap.set(subjectId, {
        name: stmt.subject.name,
        type: stmt.subject.type,
      });
    }

    // Add object node (if it has a name)
    if (stmt.object.name) {
      const objectId = `${stmt.object.type}:${stmt.object.name}`;
      if (!nodesMap.has(objectId)) {
        nodesMap.set(objectId, {
          name: stmt.object.name,
          type: stmt.object.type,
        });
      }

      // Add link
      links.push({
        source: subjectId,
        target: objectId,
        predicate: stmt.predicate,
      });
    }
  }

  const nodes = Array.from(nodesMap.entries()).map(([id, data]) => ({
    id,
    ...data,
  }));

  return { nodes, links };
}
