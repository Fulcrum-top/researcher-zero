import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from state import Summary

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "temperature", "api_key", "base_url", "model_provider"),
)

model_config = {
        "model": "kimi-k2-0905-preview",
        "model_provider": "openai",
        "max_tokens": 10000,
        "temperature": 0.0,
        "api_key": "sk-lCo4h9KTnHlXRym3eCQ1L4MsMeo9ke5Ij7LDePmiZeuwZ4Xh",
        "base_url": "https://api.moonshot.cn/v1"
    }

model = configurable_model.with_structured_output(Summary).with_config(model_config)

system_prompt = """
You are a helpful assistant that can summarize content.
"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="最近产品&UI小组在讨论如何从产品侧建立一个体系去评估MGX，除了模型强不强，效果好不好，还应该从哪些维度评估MGX，因为是探索性的工作，我在思考的过程中也感到有一些不确定性，所以和大家分享一下想法集思广益： 11月，Lovable 的增长负责人Elena提到，即便Lovable的ARR 已经超过了1亿美金，他们仍然没有找到PMF（Product Market Fit）。我们的用户画像调研参考了Lovable的做法，得到了比较稳定的用户画像，也在近期的挖掘中找到了不同画像角色的需求，而他们的需求是相对明确的。鉴于vibe coding 的市场是巨大的，用户画像是稳定的，而需求也是明确的，那为什么Lovable的PMF 每周都在变？如果画像稳定，市场巨大，需求明确，但是PMF 每周都在变，那么一定是在Fitting 上出了问题： PMF 会随着功能迭代变化，当工具能力边界不断扩张时，你每周都在匹配不同的 persona 子集，例如： 当visual editor 更稳定，PMF 就向designer / 个人展示页/ 形象类页面滑动； 当交易类链路被打通，PMF 就向高价值电商用户滑动； 用户的问题、需求、画像明确，但是用户自己不知道：实现某个需求需要什么后端/技术栈/能力/知识，所以对于用户而言，实现路径是不明确的，是否fit不是靠某个单一功能决定的，而是靠功能之间互相配合，形成的链路决定的； AI 工具本来就有随机性，用户的query也很随机，随机+随机，自然很难稳定。AI 工具很自由，但同时也带来了场景的模糊性。 所以，Lovable每周的PMF都在变这一点，也许说明的并不是vibe coding工具还没有找到市场和用户，而是：传统的PMF是静态的，但是vibe coding 的PMF 是动态的，会随着模型能力、功能迭代、A的随机性产生滑动。 鉴于PMF每周都会滑动，从产品出发的评测体系，关注重点并不是模型够不够强，甚至某个功能好不好用。而是 MGX 的可扩展能力边界在哪。没有这个框架，盲目地从四面八方扩张功能，会有如下问题 应用场景混乱； 用户体验不佳； 做了100个点，但是每个点只有20%的完成度； 没有结构、盲目扩张，就是一场功能堆积战争。如果不知道能力边界在哪，就会被竞品和用户多样的需求拖着走，陷入功能内卷循环。这个评测体系应该考虑的是：清楚知道 MGX 该做什么、不该做什么、能做什么、不能做什么。 所以，产品侧的评测体系考虑的是： 这18类场景共享的链路是什么？哪一条路是能跑通的？哪条跑不通，卡在哪？ 哪一条链路带来最大的收益？ 用户在高收益链路上的体验如何？ 总而言之，产品侧评测体系的真正目的是：MGX 的能力边界在哪里，以及哪些能力能复用、扩展、沉淀、带来营收")
]

response = asyncio.run(model.ainvoke(messages))
print(response)
breakpoint()