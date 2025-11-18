# 公開前チェックリスト

このリポジトリをpublicで公開する前に、以下の項目を確認してください。

## ✅ 完了した項目

- [x] LICENSEファイルの作成（MIT）
- [x] package.jsonにrepositoryフィールドを追加
- [x] .npmignoreファイルの作成
- [x] GitHub ActionsのCI/CD設定
- [x] セキュリティチェック（機密情報の確認）

## 📝 公開前に確認・更新が必要な項目

### 1. package.jsonの更新

以下のフィールドを実際の情報に更新してください：

```json
{
  "author": "あなたの名前 <your.email@example.com>",
  "repository": {
    "type": "git",
    "url": "https://github.com/YOUR_USERNAME/numopt-js.git"
  },
  "bugs": {
    "url": "https://github.com/YOUR_USERNAME/numopt-js/issues"
  },
  "homepage": "https://github.com/YOUR_USERNAME/numopt-js#readme"
}
```

**現在の状態**: `YOUR_USERNAME` を実際のGitHubユーザー名に置き換える必要があります。

### 2. GitHubリポジトリの作成

1. GitHubで新しいリポジトリを作成
2. リポジトリ名を `numopt-js` にする（またはpackage.jsonのnameと一致させる）
3. リポジトリをpublicに設定

### 3. リモートリポジトリの設定

```bash
# 既存のリモートを確認
git remote -v

# リモートが設定されていない場合、追加
git remote add origin https://github.com/YOUR_USERNAME/numopt-js.git

# または、既存のリモートを更新
git remote set-url origin https://github.com/YOUR_USERNAME/numopt-js.git
```

### 4. 初回コミットとプッシュ

```bash
# 変更をステージング
git add .

# コミット
git commit -m "Initial commit: Add numopt-js library"

# メインブランチにプッシュ
git push -u origin main
```

### 5. GitHub Actionsの動作確認

プッシュ後、GitHubのActionsタブでCI/CDが正常に動作するか確認してください。

## 🚀 段階的な公開戦略（推奨）

まだ自分でも十分に試せていない場合は、以下の段階的なアプローチを推奨します：

### ステップ1: GitHubに公開（テスト段階）

まずはGitHubに公開して、自分で実際に使ってみましょう。

**メリット:**
- コードは公開されるが、npmには公開されない
- 自分で実際に使ってみることができる
- 問題があれば修正してからnpmに公開できる
- 他の人もGitHubから直接インストールできる

**GitHubから直接インストールする方法:**
```bash
# 他のプロジェクトから
npm install github:YOUR_USERNAME/numopt-js

# または特定のブランチ/タグ
npm install github:YOUR_USERNAME/numopt-js#main
npm install github:YOUR_USERNAME/numopt-js#v0.1.0
```

### ステップ2: 自分で実際に使ってみる

```bash
# 別のプロジェクトで試す
cd ../my-test-project
npm install github:YOUR_USERNAME/numopt-js

# または、ローカルで開発しながら使う
npm install ../numopt-js
```

実際に使ってみて：
- [ ] 期待通りに動作するか確認
- [ ] パフォーマンスに問題がないか確認
- [ ] エッジケースで問題が起きないか確認
- [ ] ドキュメントが十分か確認
- [ ] APIが使いやすいか確認

### ステップ3: npmにスコープ付きで公開（オプション）

十分に試せたら、npmにスコープ付きパッケージとして公開することもできます：

```json
{
  "name": "@your-username/numopt-js"
}
```

**メリット:**
- パッケージ名の競合を避けられる
- まだ実験段階であることを示せる
- 後で通常のパッケージ名に変更できる

**インストール方法:**
```bash
npm install @your-username/numopt-js
```

### ステップ4: npmに正式公開（準備ができたら）

十分に試して、問題がないと確信できたら正式にnpmに公開：

```bash
# パッケージ名を確認（競合がないか）
npm view numopt-js

# ビルド
npm run build

# 公開
npm publish
```

## 📦 npm公開（準備ができたら）

npmに公開する場合は、以下の手順を実行してください。

### 1. npmアカウントの準備

```bash
# npmにログイン
npm login

# アカウント情報を確認
npm whoami
```

### 2. パッケージ名の確認

`numopt-js` という名前が既に使用されている可能性があります。確認してください：

```bash
npm view numopt-js
```

もし使用されていた場合は、`package.json`の`name`フィールドを変更する必要があります。

### 3. ビルドの確認

```bash
# クリーンビルド
npm run clean
npm run build

# ビルド成果物の確認
ls -la dist/
```

### 4. 公開前のテスト

```bash
# テストの実行
npm test

# 例の実行確認
npm run example:gradient
```

### 5. バージョンの確認

現在のバージョンは `0.1.0` です。必要に応じて更新してください：

```bash
# バージョンを更新（例：0.1.0 → 0.1.1）
npm version patch

# または手動でpackage.jsonを編集
```

### 6. npm公開

```bash
# 公開（初回）
npm publish

# 公開前に確認したい場合（dry-run）
npm publish --dry-run
```

### 7. 公開後の確認

```bash
# 公開されたパッケージを確認
npm view numopt-js

# インストールテスト
npm install numopt-js
```

## 🔒 セキュリティに関する注意事項

- ✅ `.env`ファイルは`.gitignore`に含まれています
- ✅ `node_modules/`は`.gitignore`に含まれています
- ✅ 機密情報（APIキー、パスワードなど）はコードに含まれていません
- ⚠️ 公開前に、`.gitignore`に追加すべきファイルがないか再確認してください

## 📚 公開後の推奨事項

1. **GitHubリリースの作成**: バージョンタグを作成してリリースノートを追加
2. **バッジの追加**: READMEにCI/CDのステータスバッジを追加
3. **ドキュメントの充実**: 必要に応じて追加のドキュメントを作成
4. **Issueテンプレート**: GitHubのIssueテンプレートを作成
5. **CONTRIBUTING.md**: コントリビューションガイドの作成

## 🎯 推奨される流れ

1. **まずGitHubに公開** → 自分で使ってみる
2. **実際のプロジェクトで試す** → 問題がないか確認
3. **十分に試せたら** → npmに公開を検討

この段階的なアプローチにより、リスクを最小限に抑えながら公開できます。
