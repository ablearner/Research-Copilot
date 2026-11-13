-- Run only after restoring an offline snapshot into a fresh broker.
-- Consumers fence obsolete task generations and cleanup must be idempotent.
UPDATE outbox_events
SET published_at = NULL, next_attempt_at = 0
WHERE kind IN ('task.dispatch', 'document.cleanup');
