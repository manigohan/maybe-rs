# maybe-rs 🚀

> A blazingly fast, memory-safe rewrite of the classic Unix `yes` command, in-between!

[![Made with Rust](https://img.shields.io/badge/Made%20with-Rust-orange.svg)](https://www.rust-lang.org/)
[![Memory Safety](https://img.shields.io/badge/Memory-Safe-green.svg)](https://www.rust-lang.org/)
[![Zero Cost](https://img.shields.io/badge/Abstractions-Zero%20Cost-blue.svg)](https://www.rust-lang.org/)
[![Blazing Fast](https://img.shields.io/badge/Speed-Blazing%20Fast-red.svg)](https://www.rust-lang.org/)

## Why rewrite `yes` in Rust? 🤔

Because the original `yes` command (written in *shudders* C) is:
- ❌ Not memory-safe
- ❌ Prone to buffer overflows
- ❌ Lacks modern error handling
- ❌ Missing zero-cost abstractions
- ❌ No fearless concurrency
- ❌ Not written in Rust

## Features ✨

- 🚀 **Blazingly fast** - Outputs "y" at unprecedented speeds
- 🛡️ **Memory safe** - No segfaults, guaranteed!
- ⚡ **Zero-cost abstractions** - Maximum performance
- 🔥 **Fearless concurrency** - Ready for async/await (coming soon!)
- 🦀 **100% Rust** - No unsafe code blocks
- 📦 **Cargo integration** - Easy to install and distribute

## Installation 📦

```bash
cargo install maybe-rs
```

Or build from source:

```bash
git clone https://github.com/manigohan/maybe-rs
cd maybe-rs
cargo build --release
```

## Usage 💻

Just like the original `yes`, but better:

```bash
# Output "y" forever (blazingly fast!)
maybe-rs

# Output custom string forever (memory-safe!)
maybe-rs "hello rust"
```

## Benchmarks 📊

| Command     | Language | Lines of Code | Memory Safety | Speed            |
| ----------- | -------- | ------------- | ------------- | ---------------- |
| `yes` (GNU) | C        | ~50           | ❌             | Fast             |
| `maybe-rs`    | Rust 🦀   | ~1302         | ✅             | **BLAZING FAST** |

*Benchmarks conducted on my laptop*

## Contributing 🤝

We welcome contributions! Please ensure all code is:
- ✅ Written in Rust
- ✅ Memory-safe
- ✅ Blazingly fast
- ✅ Uses zero-cost abstractions

## Share the Rust Revolution 🦀

Think this project deserves more recognition? **Share it on Hacker News!**

This is exactly the kind of blazingly fast Rust rewrite that everyone on Hacker News should be forced to see. Help spread awareness about it.

**Suggested HN title:**
- "maybe-rs: A blazingly fast, memory-safe rewrite of Unix 'yes' WRITTEN IN RUST 🦀"
- "1302 lines of Rust vs 50 lines of C: The future of systems programming"

*Because if it's not written in Rust, it's not worth using.* 🦀

---

**⚡ Powered by Rust ⚡**
