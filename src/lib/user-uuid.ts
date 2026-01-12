/**
 * Get or create a persistent user UUID from localStorage.
 * Used to track corrections from the same user.
 */
export function getUserUuid(): string {
  if (typeof window === 'undefined') {
    return '';
  }

  const key = 'statement-extractor-user-uuid';
  let uuid = localStorage.getItem(key);

  if (!uuid) {
    uuid = crypto.randomUUID();
    localStorage.setItem(key, uuid);
  }

  return uuid;
}
