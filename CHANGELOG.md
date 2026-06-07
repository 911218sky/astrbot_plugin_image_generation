# 更新日誌

## v1.0.6 - 2026-06-07

- 參考 LivingMemory 的指令組寫法，將生圖、任務、模型與預設管理統一整理到 `/img` 指令組。
- 新增 `/img gen`、`/img tasks`、`/img model`、`/img preset`、`/img help`，避免管理指令分散在 `/img_model`、`/img_tasks`、`/preset`。

## v1.0.5 - 2026-05-08

- LLM Tool 啟動圖生圖任務時，通知訊息現在會顯示參考圖數量。

## v1.0.4 - 2026-05-07

- OpenAI 生圖流程改為僅保留 GPT Image 系列
- 移除內建預設提示詞
- 補齊繁體中文介面、提示訊息與設定說明
- 強化 LLM Tool 說明與系統提示，讓模型更容易自動呼叫生圖工具
- LLM Tool 現在會依自然語言內容主動判斷較合適的寬高比與解析度
- 新增生成開始通知與失敗通知
- 補強請求異常時的失敗回覆，例如 `Server disconnected` 也會通知使用者
- 新增 `/img_tasks` 指令，可查看目前正在進行中的生圖任務
- 新增成功耗時提示開關，可在 Web 設定中控制
- 重試次數改為可設定，最高 5 次，並補充終端日誌提示
- 外掛識別名稱改為 `astrbot_plugin_image_generation_911218sky`，避免與原作者同名外掛發生更新比對衝突
- 新增會話黑名單
- 新增稽核白名單會話
- 新增安全稽核功能
- 新增 Grok AI 適配器

## 致謝

本插件基於原作者 [Railgun19457](https://github.com/railgun19457) 的
[`astrbot_plugin_image_generation`](https://github.com/railgun19457/astrbot_plugin_image_generation)
進行修改與延伸。
