// src/hooks/useSSE.ts
export function useSSE() {
    const ctrl = new AbortController();
    const start = (prompt, onToken, onDone) => {
        fetch("http://localhost:8000/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ prompt }),
            signal: ctrl.signal,
        })
            .then((res) => {
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            function pump() {
                return reader.read().then(({ value, done }) => {
                    if (done)
                        return onDone();
                    const chunk = decoder.decode(value);
                    const lines = chunk.split("\n");
                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            const data = line.slice(6);
                            if (data === "[DONE]")
                                return onDone();
                            try {
                                const { token } = JSON.parse(data);
                                onToken(token);
                            }
                            catch { }
                        }
                    }
                    return pump();
                });
            }
            pump();
        })
            .catch(() => { });
        return () => ctrl.abort();
    };
    return { start };
}
