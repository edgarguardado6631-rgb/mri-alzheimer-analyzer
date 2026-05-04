import {
  Header,
  HeaderContainer,
  HeaderName,
  HeaderMenuButton,
  HeaderGlobalBar,
  HeaderGlobalAction,
  SkipToContent,
  SideNav,
  SideNavItems,
  SideNavLink,
} from '@carbon/react';
import { Settings, UserAvatar, Home, Analytics, SettingsAdjust } from '@carbon/icons-react';
import { useNavigate, useLocation } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/', label: 'Home', Icon: Home },
  { path: '/visualization', label: 'Data', Icon: Analytics },
  { path: '/settings', label: 'Settings', Icon: SettingsAdjust },
];

const AppHeader = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <>
      <HeaderContainer
        render={({ isSideNavExpanded, onClickSideNavExpand }: { isSideNavExpanded: boolean; onClickSideNavExpand: () => void }) => (
          <>
            <Header aria-label="NeuroScan AI — MRI Analysis">
              <SkipToContent />
              <HeaderMenuButton
                aria-label={isSideNavExpanded ? 'Close menu' : 'Open menu'}
                onClick={onClickSideNavExpand}
                isActive={isSideNavExpanded}
              />
              <HeaderName href="#" prefix="" onClick={(e) => { e.preventDefault(); navigate('/'); }}>
                <img
                  src="/neuroscan-icon.png"
                  alt="NeuroScan AI"
                  width="24"
                  height="24"
                  style={{ marginRight: '0.5rem', verticalAlign: 'middle', borderRadius: '4px' }}
                />
                NeuroScan <strong style={{ fontWeight: 300 }}>AI</strong>
              </HeaderName>

              <HeaderGlobalBar>
                <HeaderGlobalAction aria-label="Settings" onClick={() => navigate('/settings')}>
                  <Settings size={20} />
                </HeaderGlobalAction>
                <HeaderGlobalAction aria-label="User profile">
                  <UserAvatar size={20} />
                </HeaderGlobalAction>
              </HeaderGlobalBar>
            </Header>

            <SideNav
              aria-label="Side navigation"
              expanded={isSideNavExpanded}
              isPersistent={false}
              onOverlayClick={onClickSideNavExpand}
            >
              <SideNavItems>
                <SideNavLink
                  renderIcon={Home}
                  isActive={location.pathname === '/'}
                  onClick={() => { navigate('/'); onClickSideNavExpand(); }}
                >
                  Welcome
                </SideNavLink>
                <SideNavLink
                  renderIcon={Analytics}
                  isActive={location.pathname === '/visualization'}
                  onClick={() => { navigate('/visualization'); onClickSideNavExpand(); }}
                >
                  Data Visualization
                </SideNavLink>
                <SideNavLink
                  renderIcon={SettingsAdjust}
                  isActive={location.pathname === '/settings'}
                  onClick={() => { navigate('/settings'); onClickSideNavExpand(); }}
                >
                  Settings
                </SideNavLink>
              </SideNavItems>
            </SideNav>
          </>
        )}
      />

      {/* Bottom navigation — visible only on mobile */}
      <nav className="bottom-nav" aria-label="Mobile navigation">
        <div className="bottom-nav-items">
          {NAV_ITEMS.map(({ path, label, Icon }) => (
            <button
              key={path}
              className={`bottom-nav-item${location.pathname === path ? ' active' : ''}`}
              onClick={() => navigate(path)}
              aria-label={label}
            >
              <Icon size={20} />
              {label}
            </button>
          ))}
        </div>
      </nav>
    </>
  );
};

export default AppHeader;
