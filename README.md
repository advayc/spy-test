# Spy Test Suite

A comprehensive monorepo for testing, benchmarking, and analyzing the Spy game algorithms with a focus on fairness evaluation.

## 🎯 Overview

This monorepo contains specialized packages for:
- **Fairness Testing**: Statistical analysis of spy selection algorithms
- **Visualization**: Professional plots and charts for test results
- **Benchmarking**: Performance profiling and optimization
- **Integration**: End-to-end testing framework

## 📊 Key Results

**Your Fisher-Yates Implementation**: **97.7/100 Fairness Score** ⭐⭐⭐⭐⭐

## 🚀 Quick Start

```bash
# Install dependencies
make install

# Run all tests
make test

# Run fairness analysis
cd packages/fairness-tests
python -m pytest tests/ -v
```

## 📁 Structure

```
spy-test/
├── packages/
│   ├── fairness-tests/     # Core fairness algorithms (97.7/100 score)
│   ├── visualization/      # Plot generation & analysis
│   ├── benchmark/          # Performance profiling
│   └── integration/        # End-to-end testing
├── docs/                   # Comprehensive documentation
├── .github/workflows/      # CI/CD pipelines
└── scripts/               # Utility scripts
```

## 🎮 Fairness Analysis

Your exact spy selection algorithm has been validated with:
- **Fairness Score**: 97.7/100 (Excellent)
- **Statistical Tests**: Chi-square, KS-test, Monte Carlo
- **Performance**: O(n) time complexity
- **Edge Cases**: Comprehensive coverage

## 📈 Features

- **Statistical Validation**: Rigorous mathematical testing
- **Performance Monitoring**: Memory and time profiling
- **Visualization**: Publication-quality charts
- **CI/CD**: Automated testing and deployment
- **Documentation**: Comprehensive guides and examples

## 🛠️ Development

### Prerequisites
- Python 3.9+
- Node.js 16+ (for monorepo management)
- Git

### Commands
```bash
make install    # Install all dependencies
make test       # Run test suite
make lint       # Code quality checks
make format     # Format code
make docs       # Generate documentation
```

## 📚 Documentation

- [Fairness Analysis](./docs/fairness-analysis.md)
- [Performance Optimization](./docs/performance-optimization.md)
- [Testing Strategies](./docs/testing-strategies.md)
- [Visualization Guide](./docs/visualization-guide.md)
- [Examples](./docs/examples/)

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🔗 Related

- **Main App**: [https://github.com/advayc/spy](https://github.com/advayc/spy)
- **Web App**: [https://github.com/advayc/spy-web](https://github.com/advayc/spy-web)