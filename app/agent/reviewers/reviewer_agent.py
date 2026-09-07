"""
结果评审智能体配置模块

将 app/prompt/prompts.yml 中的 reviewer 配置组装成
DeepAgents 可识别的字典式智能体。
第一阶段通过 subagent 机制接入主智能体，由主智能体在获得任务结果后
根据 description 调用 Reviewer 对结果进行质量检查。
"""

from app.agent.prompts import sub_agents_content
# Reviewer 的核心字段来自 YAML，便于后续只修改配置即可调整审核触发描述和评审规则
# tools 为空，表示 Reviewer 当前只负责基于已有结果进行评估，不直接调用外部工具
reviewer_agent = {
    "name": sub_agents_content["reviewer"]["name"],
    "description": sub_agents_content["reviewer"]["description"],
    "system_prompt": sub_agents_content["reviewer"]["system_prompt"],
    "tools":[],
}
