"""One native browser action per call, through the existing approval gate."""
from copy import deepcopy
import json
from time import monotonic

from agent_skills.web_research.providers import validate_url
from tool_framework.i_tool import ITool, PreparedAction, ToolPolicy, ToolResult
from tool_framework.approval import sanitized_approval_arguments

MUTATIONS = {'click', 'fill', 'press'}
# Navigation sends the URL (and session cookies) to a site, so a
# prompt-injected open could exfiltrate data through the query string.
APPROVAL_REQUIRED = MUTATIONS | {'open'}


class BrowserTool(ITool):
    def __init__(self, session):
        self.session = session
        common = {'action': {'type': 'string', 'enum': ['open', 'snapshot', 'extract', 'click', 'fill', 'press', 'scroll', 'close']},
                  'url': {'type': 'string', 'maxLength': 4000}, 'ref': {'type': 'string', 'maxLength': 100},
                  'text': {'type': 'string', 'maxLength': 4000},
                  'key': {'type': 'string', 'enum': ['Enter', 'Escape', 'Space']},
                  'pixels': {'type': 'integer', 'minimum': -2000, 'maximum': 2000}}
        super().__init__('browser', 'Operate the owned browser. Load browser skill. Use fresh snapshot refs; open/click/fill/press require approval.', [],
                         input_schema={'type': 'object', 'properties': common, 'required': ['action'], 'additionalProperties': False})

    def validate_parameters(self, args):
        super().validate_parameters(args)
        action = args['action']
        required = {'open': {'url'}, 'click': {'ref'}, 'fill': {'ref', 'text'}, 'press': {'ref', 'key'}}.get(action, set())
        allowed = {'open': {'url'}, 'click': {'ref'}, 'fill': {'ref', 'text'}, 'press': {'ref', 'key'},
                   'extract': {'ref'}, 'scroll': {'pixels'}}.get(action, set()) | {'action'}
        if not required.issubset(args) or set(args) - allowed:
            raise ValueError(f'Invalid arguments for browser {action}')
        if action == 'open':
            validate_url(args['url'])

    def policy_for(self, args):
        mutation = args['action'] in MUTATIONS
        reason = ('Opening this URL sends it, including any query data, to the website.'
                  if args['action'] == 'open' else 'The page may submit data or change external state.')
        return ToolPolicy(mutates_external=mutation, mutates_local=not mutation,
                          requires_approval=args['action'] in APPROVAL_REQUIRED, can_parallel=False,
                          default_timeout_seconds=self.session.settings.action_timeout_seconds + 75,
                          max_output_chars=20000, approval_reason=reason)

    def prepare_action(self, args):
        self.validate_input(args)
        args = deepcopy(args)
        binding = self.session.binding(args['ref']) if args.get('ref') else None
        # Keep ordinary text reviewable; use the shared credential redaction and
        # private binding rather than replacing every filled value with a hash.
        scope = sanitized_approval_arguments({**args, **(binding or {})})
        if args['action'] == 'open':
            # The full destination is what the user approves; redaction could
            # hide exactly the data being sent.
            scope['url'] = args['url']
        async def execute():
            if binding is not None and self.session.binding(args.get('ref')) != binding:
                raise ValueError('Browser state changed; take a new snapshot and obtain fresh approval')
            return await self._run(args, binding)
        return PreparedAction(self.name, args, self.policy_for(args), scope, execute)

    async def run(self, args):
        return await self.prepare_action(args).execute()

    async def _run(self, args, binding):
        start = monotonic()
        try:
            result = await self.session.execute(args, binding)
            metadata = {'capability': 'browser', 'action': args['action'], 'elapsed_ms': round((monotonic()-start)*1000)}
            if args['action'] in MUTATIONS:
                metadata['confirmed_changes'] = [f"Browser {args['action']} dispatched to the observed target; verify the resulting page before claiming task completion."]
            return ToolResult(self.name, args, json.dumps(result, ensure_ascii=False), metadata=metadata)
        except ValueError as error:
            metadata = {'not_executed': getattr(error, 'not_executed', True),
                        'error_category': 'browser', 'action': args['action']}
            if not metadata['not_executed'] and args['action'] in MUTATIONS:
                metadata['uncertain_changes'] = [f"Browser {args['action']} may have changed external state; verify before repeating it."]
            return ToolResult(self.name, args, str(error), is_error=True, metadata=metadata)
