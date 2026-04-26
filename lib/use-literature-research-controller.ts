"use client";

import { buildResearchDerivedState } from "./literature-research-controller/derived";
import { useResearchImportFlow } from "./literature-research-controller/import-flow";
import { buildLiteratureResearchControllerResult } from "./literature-research-controller/public-api";
import { useResearchQaFigureFlow } from "./literature-research-controller/qa-figure-flow";
import { useResearchConversationSession } from "./literature-research-controller/session";
import { useResearchControllerState } from "./literature-research-controller/state";
import { useResearchRuntime } from "./literature-research-controller/runtime";
import { useResearchTodoFlow } from "./literature-research-controller/todo-flow";

export function useLiteratureResearchController() {
  const controller = useResearchControllerState();
  const session = useResearchConversationSession(controller);
  const runtime = useResearchRuntime(controller, session);
  const derived = buildResearchDerivedState(controller);
  const importFlow = useResearchImportFlow(controller, session, runtime);
  const qaFigureFlow = useResearchQaFigureFlow(controller, session, derived, runtime);
  const todoFlow = useResearchTodoFlow(controller, session, runtime);

  return buildLiteratureResearchControllerResult(
    controller,
    session,
    runtime,
    derived,
    importFlow,
    qaFigureFlow,
    todoFlow
  );
}
